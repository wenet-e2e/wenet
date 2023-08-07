# Copyright (c) 2023 ASLP@NWPU (authors: Kaixun Huang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License. 


import torch
import torch.nn as nn
from typing import Tuple
from wenet.transformer.attention import MultiHeadedAttention


class BLSTM(torch.nn.Module):
    """
    """

    def __init__(self, 
                 vocab_size, 
                 embedding_size, 
                 num_layers, 
                 dropout=0.0):
        super(BLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.word_embedding = torch.nn.Embedding(
            self.vocab_size, self.embedding_size)

        self.sen_rnn = torch.nn.LSTM(input_size=self.embedding_size,
                                     hidden_size=self.embedding_size,
                                     num_layers=num_layers,
                                     dropout=dropout,
                                     batch_first=True,
                                     bidirectional=True)

    def forward(self, sen_batch, sen_lengths):
        sen_batch = torch.clamp(sen_batch, 0)
        sen_batch = self.word_embedding(sen_batch)
        pack_seq = torch.nn.utils.rnn.pack_padded_sequence(
            sen_batch, sen_lengths.to('cpu').type(torch.int32),
            batch_first=True, enforce_sorted=False)
        _, last_state = self.sen_rnn(pack_seq)
        laste_h = last_state[0]
        laste_c = last_state[1]
        state = torch.cat([laste_h[-1, :, :], laste_h[0, :, :],
                          laste_c[-1, :, :], laste_c[0, :, :]], dim=-1)
        return state


class ContextModule(torch.nn.Module):
    """
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        encoder_layers: int = 2,
        attention_heads: int = 4,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.encoder_layers = encoder_layers
        self.vocab_size = vocab_size
        self.attention_heads = attention_heads
        self.dropout_rate = dropout_rate

        self.context_extractor = BLSTM(self.vocab_size, self.embedding_size,
                                       self.encoder_layers)
        self.context_encoder = nn.Sequential(
            nn.Linear(self.embedding_size * 4, self.embedding_size),
            nn.LayerNorm(self.embedding_size)
        )

        self.biasing_layer = MultiHeadedAttention(
            n_head=self.attention_heads,
            n_feat=self.embedding_size,
            dropout_rate=self.dropout_rate
        )

        self.combiner = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.LayerNorm(self.embedding_size)
        )
        self.norm_aft_combiner = nn.LayerNorm(self.embedding_size)

        self.context_decoder = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.LayerNorm(self.embedding_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.embedding_size, self.vocab_size),
        )

        self.bias_loss = torch.nn.CTCLoss(reduction="sum")

    def forward_context_emb(self, context_list, context_lengths) -> torch.Tensor:
        context_emb = self.context_extractor(context_list, context_lengths)
        context_emb = self.context_encoder(context_emb.unsqueeze(0))
        return context_emb

    def forward(self, context_emb, encoder_out) -> Tuple[torch.Tensor, torch.Tensor]:
        context_emb = context_emb.expand(encoder_out.shape[0],-1,-1)
        context_emb, _ = self.biasing_layer(encoder_out, context_emb,
                                            context_emb)
        bias_out = self.context_decoder(context_emb)
        encoder_bias_out = self.norm_aft_combiner(encoder_out +
                                self.combiner(context_emb))
        return encoder_bias_out, bias_out