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

import random
from typing import Dict, Optional

import torch
import torchaudio
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from encodec import EncodecModel

from wenet.utils.common import (IGNORE_ID, add_sos_eos, th_accuracy)
from wenet.utils.class_utils import WENET_EMB_CLASSES
from wenet.utils.mask import make_pad_mask


class VallE(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 tie_word_embedding: bool = True,
                 num_blocks: int = 12,
                 attention_heads: int = 16,
                 attention_dim: int = 1024,
                 linear_units: int = 4096,
                 dropout_rate: float = 0.1,
                 ctc_weight: float = 0.3):
        super().__init__()
        self.audio_size = 1024 + 1  # 1 is last one <sos/eos>
        self.num_quantizer = 8
        self.text_sos = 2
        self.text_eos = 2
        self.audio_sos = 1024
        self.audio_eos = 1024
        self.ignore_id = IGNORE_ID
        self.nhead = attention_heads
        self.ar_decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=attention_dim,
                                       nhead=self.nhead,
                                       dim_feedforward=linear_units,
                                       batch_first=True),
            num_layers=num_blocks,
            norm=nn.LayerNorm(attention_dim, eps=1e-5),
        )
        self.nar_decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=attention_dim,
                                       nhead=self.nhead,
                                       dim_feedforward=linear_units,
                                       batch_first=True),
            num_layers=num_blocks,
            norm=nn.LayerNorm(attention_dim, eps=1e-5),
        )
        self.ar_text_embedding = nn.Sequential(
            nn.Embedding(vocab_size, attention_dim),
            WENET_EMB_CLASSES['abs_pos'](attention_dim, 0.1),
        )
        self.nar_text_embedding = nn.Sequential(
            nn.Embedding(vocab_size, attention_dim),
            WENET_EMB_CLASSES['abs_pos'](attention_dim, 0.1),
        )
        self.audio_embedding = nn.ModuleList([
            nn.Sequential(
                nn.Embedding(self.audio_size, attention_dim),
                WENET_EMB_CLASSES['abs_pos'](attention_dim, 0.1),
            ) for i in range(self.num_quantizer)
        ])
        self.projection = nn.ModuleList([
            nn.Linear(attention_dim, self.audio_size)
            for i in range(self.num_quantizer)
        ])
        if tie_word_embedding:
            for i in range(self.num_quantizer):
                self.projection[i].weight = self.audio_embedding[i][0].weight
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
        audio = pad_sequence(audio, batch_first=True, padding_value=0)
        text_mask = make_pad_mask(text_lengths)
        text = text.masked_fill(text_mask, self.text_eos)
        text = F.pad(text, (1, 1), value=self.text_eos)  # eos same as sos
        text_lengths = text_lengths + 2
        text_pad_mask = make_pad_mask(text_lengths)
        audio_pad_mask = make_pad_mask(audio_lengths + 1)  # add sos/eos
        text_audio_pad_mask = torch.concat([text_pad_mask, audio_pad_mask],
                                           dim=1)
        text_len, audio_len = text.size(1), audio.size(1) + 1
        text_audio_len = text_len + audio_len
        batch_size = text.size(0)

        # 2-1. AR decoder branch
        ar_text_emb, _ = self.ar_text_embedding(text)
        ar_audio_in, ar_audio_out = add_sos_eos(audio[:, :, 0], self.audio_sos,
                                                self.audio_eos, self.ignore_id)
        ar_audio_emb, _ = self.audio_embedding[0](ar_audio_in)
        ar_text_audio_emb = torch.concat([ar_text_emb, ar_audio_emb], dim=1)
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
        text_audio_attn_mask = torch.concat([text_attn_mask, audio_attn_mask],
                                            dim=0)
        pad_mask = text_audio_pad_mask.view(batch_size, 1, 1, text_audio_len)
        pad_mask = pad_mask.expand(-1, self.nhead, -1, -1)
        pad_mask = pad_mask.reshape(batch_size * self.nhead, 1, text_audio_len)
        text_audio_attn_mask = text_audio_attn_mask.logical_or(pad_mask)
        fmask = torch.zeros_like(text_audio_attn_mask, dtype=torch.float)
        fmask = fmask.masked_fill(text_audio_attn_mask, float('-inf'))
        ar_decoder_out = self.ar_decoder(ar_text_audio_emb, fmask)
        ar_decoder_out = self.projection[0](
            ar_decoder_out)[:, text_len:, :].contiguous()
        ar_loss = F.cross_entropy(ar_decoder_out.permute(0, 2, 1),
                                  ar_audio_out,
                                  ignore_index=self.ignore_id)
        ar_acc = th_accuracy(ar_decoder_out.view(-1, self.audio_size),
                             ar_audio_out,
                             ignore_label=self.ignore_id)
        # 2-2. NAR decoder branch, random sample one to train
        k = random.randint(1, self.num_quantizer - 1)
        nar_text_emb, _ = self.nar_text_embedding(text)
        nar_audio_in, nar_audio_out = audio[:, :, k - 1], audio[:, :, k]
        nar_audio_emb, _ = self.audio_embedding[k](nar_audio_in)
        nar_text_audio_emb = torch.concat([nar_text_emb, nar_audio_emb], dim=1)
        audio_pad_mask = make_pad_mask(audio_lengths)
        nar_audio_out = nar_audio_out.masked_fill(audio_pad_mask,
                                                  self.ignore_id)
        text_audio_mask = torch.concat([text_pad_mask, audio_pad_mask], dim=1)
        nar_decoder_out = self.nar_decoder(
            nar_text_audio_emb, src_key_padding_mask=text_audio_mask)
        nar_decoder_out = self.projection[k](
            nar_decoder_out)[:, text_len:, :].contiguous()
        nar_loss = F.cross_entropy(nar_decoder_out.permute(0, 2, 1),
                                   nar_audio_out,
                                   ignore_index=self.ignore_id)
        nar_acc = th_accuracy(nar_decoder_out.view(-1, self.audio_size),
                              nar_audio_out,
                              ignore_label=self.ignore_id)

        loss = ar_loss + nar_loss
        return {
            'loss': loss,
            'ar_loss': ar_loss,
            'nar_loss': nar_loss,
            'ar_acc': torch.tensor(ar_acc),
            'nar_acc': torch.tensor(nar_acc),
        }
