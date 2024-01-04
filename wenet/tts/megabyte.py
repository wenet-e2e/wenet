# Copyright (c) 2023 Binbin Zhang(binbzha@qq.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, Optional

import torch
import torchaudio
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from encodec import EncodecModel

from wenet.utils.common import (IGNORE_ID, th_accuracy)
from wenet.utils.class_utils import WENET_EMB_CLASSES
from wenet.utils.mask import make_pad_mask, subsequent_mask


class MegaByte(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 g_num_layers: int = 12,
                 g_nhead: int = 8,
                 g_d_model: int = 512,
                 g_dim_feedforward: int = 2048,
                 l_num_layers: int = 6,
                 l_nhead: int = 8,
                 l_d_model: int = 256,
                 l_dim_feedforward: int = 1024,
                 ctc_weight: float = 0.3):
        super().__init__()
        self.audio_size = 1024 + 1  # 1 is last one <sos/eos>
        self.num_quantizer = 8
        self.text_sos = 2
        self.text_eos = 2
        self.audio_sos = 1024
        self.audio_eos = 1024
        self.ignore_id = IGNORE_ID
        self.g_nhead = g_nhead
        assert g_d_model % self.num_quantizer == 0
        self.g_embedding_size = int(g_d_model / self.num_quantizer)
        self.g_model = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=g_d_model,
                                       nhead=self.g_nhead,
                                       dim_feedforward=g_dim_feedforward,
                                       batch_first=True),
            num_layers=g_num_layers,
            norm=nn.LayerNorm(g_d_model, eps=1e-5),
        )
        self.l_model = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=l_d_model,
                                       nhead=l_nhead,
                                       dim_feedforward=l_dim_feedforward,
                                       batch_first=True),
            num_layers=l_num_layers,
            norm=nn.LayerNorm(l_d_model, eps=1e-5),
        )
        self.g_audio_embedding = nn.Sequential(
            nn.Embedding(self.audio_size, self.g_embedding_size),
            WENET_EMB_CLASSES['abs_pos'](self.g_embedding_size, 0.1),
        )
        self.l_audio_embedding = nn.Sequential(
            nn.Embedding(self.audio_size, l_d_model),
            WENET_EMB_CLASSES['abs_pos'](l_d_model, 0.1),
        )
        self.text_embedding = nn.Sequential(
            nn.Embedding(vocab_size, g_d_model),
            WENET_EMB_CLASSES['abs_pos'](g_d_model, 0.1),
        )
        self.g2l_linear = nn.Linear(self.g_embedding_size, l_d_model)
        self.projection = nn.Linear(l_d_model, self.audio_size)
        self.codec = EncodecModel.encodec_model_24khz()
        self.codec.set_target_bandwidth(6.0)

    def forward(self, batch: dict,
                device: torch.device) -> Dict[str, Optional[torch.Tensor]]:
        text = batch['target'].to(device)
        text_lengths = batch['target_lengths'].to(device)
        wavs = batch['pcm']
        # 1. on-the-fly quantization
        audio = []
        for wav in wavs:
            wav = wav.to(device).unsqueeze(0)
            wav = torchaudio.functional.resample(wav, 16000,
                                                 self.codec.sample_rate)
            wav = wav.unsqueeze(0)
            with torch.no_grad():
                encoded_frames = self.codec.encode(wav)
            vq = encoded_frames[0][0][0].transpose(0, 1)
            audio.append(vq)
        audio_lengths = torch.tensor([x.size(0) for x in audio],
                                     dtype=torch.int32,
                                     device=device)
        audio = pad_sequence(audio,
                             batch_first=True,
                             padding_value=self.audio_eos)
        text_mask = make_pad_mask(text_lengths)
        text = text.masked_fill(text_mask, self.text_eos)
        text = F.pad(text, (1, 1), value=self.text_eos)  # eos same as sos
        text_lengths = text_lengths + 2
        text_pad_mask = make_pad_mask(text_lengths)
        audio_pad_mask = make_pad_mask(audio_lengths + 1)  # add sos
        text_audio_pad_mask = torch.concat([text_pad_mask, audio_pad_mask],
                                           dim=1)
        text_len, audio_len = text.size(1), audio.size(1) + 1
        text_audio_len = text_len + audio_len
        batch_size = text.size(0)
        # 2. Global model
        text_emb, _ = self.text_embedding(text)
        g_audio = torch.concat(
            [torch.ones_like(audio[:, :1, :]) * self.audio_sos, audio],
            dim=1)  # add sos
        g_audio_emb, _ = self.g_audio_embedding(g_audio.view(batch_size, -1))
        g_audio_emb = g_audio_emb.view(batch_size, audio_len, -1)
        text_audio_emb = torch.concat([text_emb, g_audio_emb], dim=1)
        text_attn_mask = F.pad(
            torch.zeros((text_len, text_len), dtype=torch.bool, device=device),
            (0, audio_len),
            value=True,
        )
        audio_attn_mask = F.pad(
            torch.triu(
                torch.ones(audio_len,
                           audio_len,
                           dtype=torch.bool,
                           device=device),
                diagonal=1,
            ),
            (text_len, 0),
            value=False,
        )
        attn_mask = torch.concat([text_attn_mask, audio_attn_mask], dim=0)
        pad_mask = text_audio_pad_mask.view(batch_size, 1, 1, text_audio_len)
        pad_mask = pad_mask.expand(-1, self.g_nhead, -1, -1)
        pad_mask = pad_mask.reshape(batch_size * self.g_nhead, 1,
                                    text_audio_len)
        attn_mask = attn_mask.logical_or(pad_mask)
        f_mask = torch.zeros_like(attn_mask, dtype=torch.float)
        f_mask = f_mask.masked_fill(attn_mask, float('-inf'))
        g_output = self.g_model(text_audio_emb,
                                f_mask)[:, text_len:, :].contiguous()
        g_output = g_output.view(batch_size * audio_len, self.num_quantizer,
                                 -1)
        g_logits = self.g2l_linear(g_output)
        # 3. Local model
        l_audio = torch.concat(
            [audio, torch.ones_like(audio[:, :1, :]) * self.audio_eos],
            dim=1)  # add global eos
        l_label = l_audio.masked_fill(audio_pad_mask.unsqueeze(-1),
                                      self.ignore_id)
        l_label = l_label.view(batch_size * audio_len, self.num_quantizer)
        l_audio = l_audio.view(batch_size * audio_len, self.num_quantizer)
        l_input = F.pad(l_audio[:, :-1], (1, 0),
                        value=self.audio_sos)  # add local sos
        l_input, _ = self.l_audio_embedding(l_input)
        l_input = l_input + g_logits
        mask = ~subsequent_mask(self.num_quantizer, device)
        l_logits = self.l_model(l_input, mask)
        l_logits = self.projection(l_logits)
        loss = F.cross_entropy(l_logits.permute(0, 2, 1),
                               l_label,
                               ignore_index=self.ignore_id)
        acc = th_accuracy(l_logits.view(-1, self.audio_size),
                          l_label,
                          ignore_label=self.ignore_id)
        return {
            'loss': loss,
            'acc': torch.tensor(acc),
        }

    def inference(self, audio: torch.Tensor, ref_text: torch.Tensor,
                  syn_text: torch.Tensor, device: torch.device):
        batch_size = audio.size(0)
        assert batch_size == 1
        text = torch.concat([ref_text, syn_text], dim=1)
        print(text)
        text = F.pad(text, (1, 1), value=self.text_eos)  # add sos & eos
        text_len = text.size(1)
        text_emb, _ = self.text_embedding(text)

        max_len = 75 * 1  # 2 seconds
        src_audio = audio
        # TODO(Binbin Zhang): Add cache
        for step in range(max_len):
            # Global
            g_audio = torch.concat(
                [torch.ones_like(audio[:, :1, :]) * self.audio_sos, audio],
                dim=1)  # add sos
            audio_len = g_audio.size(1)
            g_audio_emb, _ = self.g_audio_embedding(
                g_audio.view(batch_size, -1))
            g_audio_emb = g_audio_emb.view(batch_size, audio_len, -1)
            text_audio_emb = torch.concat([text_emb, g_audio_emb], dim=1)
            text_attn_mask = F.pad(
                torch.zeros((text_len, text_len),
                            dtype=torch.bool,
                            device=device),
                (0, audio_len),
                value=True,
            )
            audio_attn_mask = F.pad(
                torch.triu(
                    torch.ones(audio_len,
                               audio_len,
                               dtype=torch.bool,
                               device=device),
                    diagonal=1,
                ),
                (text_len, 0),
                value=False,
            )
            attn_mask = torch.concat([text_attn_mask, audio_attn_mask], dim=0)
            g_output = self.g_model(text_audio_emb,
                                    attn_mask)[:, -1, :].contiguous()
            g_output = g_output.view(batch_size, self.num_quantizer,
                                     -1)  # 1, 8, g_emb
            g_logits = self.g2l_linear(g_output)  # 1, 8, l_d_model
            # Local
            la = [self.audio_sos]
            for i in range(self.num_quantizer):
                l_input = torch.tensor(la, dtype=torch.long,
                                       device=device).unsqueeze(0)
                l_input, _ = self.l_audio_embedding(l_input)
                l_input = l_input + g_logits[:, :i + 1, :]
                mask = ~subsequent_mask(i + 1, device)
                l_logits = self.l_model(l_input, mask)
                l_logits = self.projection(l_logits)
                pred = l_logits[0, -1, :].argmax().item()
                la.append(pred)
            print(step, la[1:])
            if self.audio_eos in la[1:]:
                break
            gen = torch.tensor(la[1:], dtype=torch.long, device=device)
            gen = gen.view(1, 1, self.num_quantizer)
            audio = torch.concat([audio, gen], dim=1)
            print(audio.size())
        return audio
