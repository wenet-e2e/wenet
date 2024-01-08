from typing import Optional, Tuple
from wenet.transformer.attention import MultiHeadedAttention
from torch import nn
import math
import torch


class MultiHeadedAttentionSANM(MultiHeadedAttention):
    """Multi-Head Attention layer.
    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
    """

    def __init__(self,
                 n_head,
                 in_feat,
                 n_feat,
                 dropout_rate,
                 kernel_size,
                 sanm_shfit=0):
        """Construct an MultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate)
        # We assume d_v always equals d_k
        # self.linear_q = nn.Linear(n_feat, n_feat)
        # self.linear_k = nn.Linear(n_feat, n_feat)
        # self.linear_v = nn.Linear(n_feat, n_feat)
        del self.linear_q, self.linear_k, self.linear_v
        self.linear_q_k_v = nn.Linear(in_feat, n_feat * 3)

        self.fsmn_block = nn.Conv1d(n_feat,
                                    n_feat,
                                    kernel_size,
                                    stride=1,
                                    padding=0,
                                    groups=n_feat,
                                    bias=False)
        # padding
        self.left_padding = (kernel_size - 1) // 2
        if sanm_shfit > 0:
            self.left_padding = self.left_padding + sanm_shfit
        self.right_padding = kernel_size - 1 - self.left_padding
        self.pad_fn = nn.ConstantPad1d((self.left_padding, self.right_padding),
                                       0.0)

    def forward_qkv(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        x = query
        b, t, _ = x.size()
        q_k_v = self.linear_q_k_v(x)
        q, k, v = torch.split(q_k_v, int(self.h * self.d_k), dim=-1)
        q = torch.reshape(q, (b, t, self.h, self.d_k)).transpose(
            1, 2)  # (batch, head, time1, d_k)
        k = torch.reshape(k, (b, t, self.h, self.d_k)).transpose(
            1, 2)  # (batch, head, time2, d_k)
        v = torch.reshape(v, (b, t, self.h, self.d_k)).transpose(
            1, 2)  # (batch, head, time2, d_k)

        return q, k, v

    def forward_fsmn(self,
                     inputs: torch.Tensor,
                     mask: torch.Tensor,
                     mask_shfit_chunk: Optional[torch.Tensor] = None):
        b, _, t, _ = inputs.size()
        inputs = inputs.transpose(1, 2).view(b, t, -1)
        if mask.size(2) > 0:  # time2 > 0
            # TODO(Mddct): make sure mask is right
            if mask_shfit_chunk is not None:
                mask = mask * mask_shfit_chunk
            mask = mask.transpose(1, 2)  # [B,T,1]
            inputs = inputs * mask
        x = inputs.transpose(1, 2)
        # x = torch.nn.functional.pad(x, (self.left_padding, self.right_padding),
        #                             value=0.0,
        #                             mode='constant')
        x = self.pad_fn(x)
        x = self.fsmn_block(x)
        x = x.transpose(1, 2)
        x += inputs
        x = self.dropout(x)
        return x * mask

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        pos_emb: torch.Tensor = torch.empty(0),
        cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
        mask_shfit_chunk: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q, k, v = self.forward_qkv(query, key, value)
        if cache.size(0) > 0:
            key_cache, value_cache = torch.split(cache,
                                                 cache.size(-1) // 2,
                                                 dim=-1)
            k = torch.cat([key_cache, k], dim=2)
            v = torch.cat([value_cache, v], dim=2)
        # NOTE(Mddct): we need know fsmn_memory's cache, but paraformer is nonstreamming
        # refactor later if streaming model is available
        new_cache = torch.cat((k, v), dim=-1)
        fsmn_memory = self.forward_fsmn(v,
                                        mask=mask_pad,
                                        mask_shfit_chunk=mask_shfit_chunk)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        att = self.forward_attention(v, scores, mask)
        return att + fsmn_memory, new_cache


class DummyMultiHeadSANM(MultiHeadedAttentionSANM):
    """A dummy multihead attention for Paraformer befroe cross attention
    """

    def __init__(self,
                 n_head,
                 in_feat,
                 n_feat,
                 dropout_rate,
                 kernel_size,
                 sanm_shfit=0):
        super().__init__(n_head, in_feat, n_feat, dropout_rate, kernel_size,
                         sanm_shfit)
        del self.linear_q_k_v
        del self.linear_out

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        pos_emb: torch.Tensor = torch.empty(0),
        cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
        mask_shfit_chunk: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        query = query * mask_pad.transpose(1, 2)
        inputs = query
        x = inputs.transpose(1, 2)
        x = self.pad_fn(x)
        # TODO(Mddct): cache here for future streaming
        cache: Optional[torch.Tensor] = None
        x = self.fsmn_block(x)
        x = x.transpose(1, 2)
        if x.size(1) != inputs.size(1):
            inputs = inputs[:, -1, :]

        x = x + inputs
        x = self.dropout(x)
        x = x * mask_pad.transpose(1, 2)
        return x, cache


class MultiHeadAttentionCross(MultiHeadedAttentionSANM):

    def __init__(self,
                 n_head,
                 in_feat,
                 n_feat,
                 dropout_rate,
                 kernel_size,
                 sanm_shfit=0,
                 target_size: Optional[int] = None):
        super().__init__(n_head, in_feat, n_feat, dropout_rate, kernel_size,
                         sanm_shfit)
        del self.linear_q_k_v
        del self.fsmn_block
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k_v = nn.Linear(
            n_feat if target_size is None else target_size, n_feat * 2)

    def forward_qkv(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # NOTE(Mddct): here value == key
        _ = value

        x = query
        b = x.size(0)
        q = self.linear_q(x)
        q_h = torch.reshape(q, (b, -1, self.h, self.d_k)).transpose(
            1, 2)  # (batch, head, time1, d_k)

        k_v = self.linear_k_v(key)
        k, v = torch.split(k_v, int(self.h * self.d_k), dim=-1)
        k_h = torch.reshape(k, (b, -1, self.h, self.d_k)).transpose(
            1, 2)  # (batch, head, time2, d_k)
        v_h = torch.reshape(v, (b, -1, self.h, self.d_k)).transpose(
            1, 2)  # (batch, head, time2, d_k)

        return q_h, k_h, v_h

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        pos_emb: torch.Tensor = torch.empty(0),
        cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
        mask_shfit_chunk: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        q, k, v = self.forward_qkv(query, key, key)
        q = q * self.d_k**(-0.5)
        scores = torch.matmul(q, k.transpose(-2, -1))

        # TODO(Mddct): support future streaming paraformer
        cache: Optional[torch.Tensor] = None
        return self.forward_attention(v, scores, mask), cache
