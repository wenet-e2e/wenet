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

import torch
import torchaudio
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from encodec import EncodecModel

from wenet.transformer.attention import MultiHeadedAttention
from wenet.transformer.embedding import PositionalEncoding
from wenet.transformer.encoder_layer import TransformerEncoderLayer
from wenet.transformer.positionwise_feed_forward import PositionwiseFeedForward

from wenet.utils.common import add_sos_eos
from wenet.utils.mask import make_pad_mask, subsequent_mask


class TransformerDecoderOnly(nn.Module):

    def __init__(
        self,
        num_layers: int = 12,
        nhead: int = 8,
        d_model: int = 512,
        dim_feedforward: int = 2048,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.encoders = nn.ModuleList([
            TransformerEncoderLayer(
                d_model,
                MultiHeadedAttention(
                    nhead,
                    d_model,
                    attention_dropout_rate,
                    key_bias=True,
                ),
                PositionwiseFeedForward(
                    d_model,
                    dim_feedforward,
                    dropout_rate,
                    activation=nn.ReLU(),
                ),
                dropout_rate,
                normalize_before=True,
            ) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model, eps=1e-5)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pos_emb: torch.Tensor,
        kv_cache: torch.Tensor = None,
    ):
        """
        Args:
            kv_cache: List[torch.Tensor], list size is num layers
        """
        cache = []
        for i, layer in enumerate(self.encoders):
            if kv_cache is None:
                x, mask, c, _ = layer(x, mask, pos_emb)
            else:
                x, mask, c, _ = layer(x, mask, pos_emb, att_cache=kv_cache[i])
            cache.append(c)
        x = self.norm(x)
        return x, cache


class VQTTS(nn.Module):

    def __init__(
        self,
        vocab_size: int = 32000,
        num_layers: int = 12,
        nhead: int = 8,
        d_model: int = 512,
        dim_feedforward: int = 2048,
        max_len=10000,
        ctc_weight: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        # 1024 for padding, 1025 for eos
        self.codebook_size = 1024 + 1
        self.code_sos = self.codebook_size - 1
        self.code_eos = self.codebook_size - 1
        self.num_codebooks = 2
        self.codec = EncodecModel.encodec_model_24khz()
        self.codec.set_target_bandwidth(1.5)
        self.text_sos = 2
        self.text_eos = 2
        self.model = TransformerDecoderOnly(
            num_layers=num_layers,
            nhead=nhead,
            d_model=d_model,
            dim_feedforward=dim_feedforward,
        )
        self.text_embedding = nn.Embedding(vocab_size, d_model)
        self.audio_embedding = nn.ModuleList([
            nn.Embedding(self.codebook_size, d_model)
            for i in range(self.num_codebooks)
        ])
        self.pos_encoding = PositionalEncoding(d_model, 0.1, max_len)
        self.output = nn.Linear(
            d_model,
            self.vocab_size + self.codebook_size * self.num_codebooks,
            bias=False,
        )
        self.ignore_id = -1

    def quantize(self, wavs, wavs_lengths, device):
        B = wavs.size(0)
        codes = []
        for i in range(B):
            wav = wavs[i, :wavs_lengths[i]].to(device).unsqueeze(0)
            wav = torchaudio.functional.resample(wav, 16000,
                                                 self.codec.sample_rate)
            wav = wav.unsqueeze(0)
            with torch.no_grad():
                encoded_frames = self.codec.encode(wav)
            vq = encoded_frames[0][0][0].transpose(0, 1)
            codes.append(vq)
        codes_lengths = torch.tensor([x.size(0) for x in codes],
                                     dtype=torch.int32,
                                     device=device)
        codes = pad_sequence(codes,
                             batch_first=True,
                             padding_value=self.code_eos)
        return codes, codes_lengths

    def codes_embedding(self, codes_in):
        # Sum all VQ embedding
        codes_emb = []
        for i in range(self.num_codebooks):
            codes_emb.append(self.audio_embedding[i](codes_in[:, :, i]))
        codes_emb = torch.stack(codes_emb, dim=3)
        codes_emb = codes_emb.sum(dim=3)  # (B, C, D)
        return codes_emb

    def compute(self, batch: dict, device: torch.device):
        '''
        Let's assume:
            B: batch_size
            T: text_length
            C: codes_length
        '''
        text = batch['target'].to(device)
        text_lengths = batch['target_lengths'].to(device)
        wavs = batch['pcm']
        wavs_lengths = batch['pcm_length']
        B = text.size(0)
        # on-the-fly quantization
        codes, codes_lengths = self.quantize(wavs, wavs_lengths, device)

        text_lengths = text_lengths + 1
        text_mask = make_pad_mask(text_lengths)  # (B, T)
        text_in, text_label = add_sos_eos(text, self.text_sos, self.text_eos,
                                          self.ignore_id)
        # [sos, codes, eos]
        codes = F.pad(codes.transpose(1, 2), (1, 1), value=self.code_sos)
        codes = codes.transpose(1, 2)
        codes_in = codes[:, :-1, :]
        codes_label = codes[:, 1:, :]
        codes_lengths = codes_lengths + 1
        codes_mask = make_pad_mask(codes_lengths)  # (B, C)
        codes_label = codes_label.masked_fill(codes_mask.unsqueeze(-1),
                                              self.ignore_id)
        codes_emb = self.codes_embedding(codes_in)
        # Mask
        token_mask = torch.cat((~text_mask, ~codes_mask),
                               dim=1).unsqueeze(1)  # (B, 1, T+C)
        ar_mask = subsequent_mask(token_mask.size(1),
                                  device).unsqueeze(0)  # (1, T+C, T+C)
        mask = token_mask & ar_mask  # (B, T+C, T+C)
        text_emb = self.text_embedding(text_in)  # (B, T, D)
        all_emb = torch.cat((text_emb, codes_emb), dim=1)  # (B, T+C, D)
        all_emb, pos_emb = self.pos_encoding(all_emb)
        output, kv_cache = self.model(all_emb, mask, pos_emb)
        logits = self.output(output)
        return logits, kv_cache, text_label, codes_label

    def forward(self, batch: dict, device: torch.device):
        logits, _, text_label, codes_label = self.compute(batch, device)
        B = text_label.size(0)
        T = text_label.size(1)
        C = codes_label.size(1)
        text_logits = logits[:, :T, :self.vocab_size]  # (B, T, ...)
        loss_text = F.cross_entropy(
            text_logits.reshape(-1, self.vocab_size),
            text_label.reshape(-1),
            ignore_index=self.ignore_id,
        )
        codes_logits = logits[:, T:, self.vocab_size:]  # (B, C, ...)
        codes_logits = codes_logits.reshape(B, C, self.num_codebooks, -1)
        loss_codes = F.cross_entropy(
            codes_logits.reshape(-1, self.codebook_size),
            codes_label.reshape(-1),
            ignore_index=self.ignore_id,
        )
        # Compute Accuracy
        pred = text_logits.argmax(2)
        correct = pred.eq(text_label)
        correct[text_label == self.ignore_id] = 0
        correct = correct.sum()
        acc_text = correct / (text_label != self.ignore_id).sum()
        _, indices = codes_logits.topk(5, dim=-1)
        correct = indices.eq(codes_label.unsqueeze(-1))
        correct[codes_label == self.ignore_id] = 0
        correct = correct.sum()
        acc_codes = correct / (codes_label != self.ignore_id).sum()

        loss = loss_text + loss_codes
        return {
            'loss': loss,
            'loss_text': loss_text,
            'loss_codes': loss_codes,
            'acc_text': acc_text,
            'acc_codes': acc_codes,
        }

    # Copy from https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L398C1-L421C22
    def sample_top_p(self, probs, p):
        """
        Perform top-p (nucleus) sampling on a probability distribution.

        Args:
            probs (torch.Tensor): Probability distribution tensor.
            p (float): Probability threshold for top-p sampling.

        Returns:
            torch.Tensor: Sampled token indices.

        Note:
            Top-p sampling selects the smallest set of tokens whose cumulative probability mass
            exceeds the threshold p. The distribution is renormalized based on the selected tokens.

        """
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token

    def infer(self,
              batch: dict,
              device: torch.device,
              temperature=1.0,
              top_p=0.9,
              repeatition_penalty=1.5):
        self.codec.eval()
        logits, kv_cache, _, _ = self.compute(batch, device)
        offset = logits.size(1)
        codes_logit = logits[:, -1, self.vocab_size:]
        # Autogressive generate
        max_steps = 200
        codes_pred = []
        for i in range(max_steps):
            codes_logit = codes_logit.reshape(self.num_codebooks,
                                              self.codebook_size)
            win_size = 16
            if len(codes_pred) < win_size:
                window = codes_pred
            else:
                window = codes_pred[len(codes_pred) - win_size:]
            # Add repeatition penalty
            if len(window) > 0 and repeatition_penalty != 1.0:
                prev_tokens = torch.cat(window, dim=1).squeeze(0)
                prev_tokens = prev_tokens.transpose(0, 1)
                score = torch.gather(codes_logit, dim=1, index=prev_tokens)
                score = torch.where(score < 0, score * repeatition_penalty,
                                    score / repeatition_penalty)
                codes_logit.scatter_(dim=1, index=prev_tokens, src=score)
            # Top-p sampling
            codes_prob = torch.softmax(codes_logit / temperature, dim=-1)
            # print('############')
            pred = self.sample_top_p(codes_prob, top_p)
            pred = pred.reshape(-1)
            print('step', i, 'prediction', pred)
            # if we get eos, done
            if (pred == self.code_eos).any():
                break
            pred = pred.unsqueeze(0).unsqueeze(0)
            codes_pred.append(pred)
            codes_emb = self.codes_embedding(pred)  # (1, 1, D)
            codes_emb, pos_emb = self.pos_encoding(codes_emb, offset)
            mask = torch.ones((1, 1, 1), dtype=torch.bool, device=device)
            output, kv_cache = self.model(codes_emb, mask, pos_emb, kv_cache)
            logits = self.output(output)
            codes_logit = logits[:, -1, self.vocab_size:]
            offset += 1
        codes_pred = torch.cat(codes_pred, dim=1)
        wav = self.codec.decode([(codes_pred.transpose(1, 2), None)])
        return wav, self.codec.sample_rate
