"""Multi-Head Attention layer definition."""

import math
from typing import Tuple

import torch
from torch import nn
from wenet.transformer.attention import MultiHeadedAttention

class ChunkAttentionWithRelativeRightContext(MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding.
    Paper: https://arxiv.org/abs/1901.02860
    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
    """
    def __init__(self, n_head, n_feat, dropout_rate):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate)
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x, left_context_size: int = 0, right_context_size: int = 0):
        """Compute relative positional encoding. The position should capture both 
        left and right context.

        Args:
            x: Input tensor (batch, head, time1, 2*time1-1+left_context_size).
                time1 means the length of query vector.
            left_context_size (int): Left context size for limited chunk context
            right_context_size (int): Right context size for limited chunk context
        Returns:
            Tensor: tensor of shape (batch, head, time1, time2)
          (note: time2 has the same value as time1, but it is for
          the key, while time1 is for the query).
        """
        (batch_size, num_heads, time1, n) = x.size()
        time2 = time1 + left_context_size + right_context_size
        batch_stride = x.stride(0)
        head_stride = x.stride(1)
        time1_stride = x.stride(2)
        n_stride = x.stride(3)
        return x.as_strided(
            (batch_size, num_heads, time1, time2),
            (batch_stride, head_stride, time1_stride - n_stride, n_stride),
            storage_offset=n_stride * (time1 - 1),
        )

    def forward(self, query: torch.Tensor,
                key: torch.Tensor, value: torch.Tensor,
                mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
                pos_emb: torch.Tensor = torch.empty(0),
                cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
                chunk_size: int = 0,
                left_context_size: int = 0,
                right_context_size: int = 0,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2), (0, 0, 0) means fake mask.
            pos_emb (torch.Tensor): Positional embedding tensor
                (#batch, time2, size).
            cache (torch.Tensor): Cache tensor (B, 1, head, cache_t, d_k * 2),
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`
            chunk_size (int): Chunk size for limited chunk context
            left_context_size (int): Left context size for limited chunk context
            right_context_size (int): Right context size for limited chunk context
        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
            torch.Tensor: Cache tensor (1, head, cache_t + time1, d_k * 2)
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`
        """
        bz = query.shape[0]
        n_feat = query.shape[2]
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        limited_context_attn = (chunk_size > 0 
                                and left_context_size > 0 
                                and right_context_size > 0)

        # NOTE(xcsong):
        #   when export onnx model, for 1st chunk, we feed
        #       cache(1, head, 0, d_k * 2) (16/-1, -1/-1, 16/0 mode)
        #       or cache(1, head, real_cache_t, d_k * 2) (16/4 mode).
        #       In all modes, `if cache.size(0) > 0` will alwayse be `True`
        #       and we will always do splitting and
        #       concatnation(this will simplify onnx export). Note that
        #       it's OK to concat & split zero-shaped tensors(see code below).
        #   when export jit  model, for 1st chunk, we always feed
        #       cache(0, 0, 0, 0) since jit supports dynamic if-branch.
        # >>> a = torch.ones((1, 2, 0, 4))
        # >>> b = torch.ones((1, 2, 3, 4))
        # >>> c = torch.cat((a, b), dim=2)
        # >>> torch.equal(b, c)        # True
        # >>> d = torch.split(a, 2, dim=-1)
        # >>> torch.equal(d[0], d[1])  # True
        if cache.size(2) > 0:
            key_cache, value_cache = torch.split(
                cache, cache.size(-1) // 2, dim=-1)
            k = torch.cat([key_cache, k], dim=2)
            v = torch.cat([value_cache, v], dim=2)

            # NOTE(xcsong): We do cache slicing in encoder.forward_chunk, since it's
            #   non-trivial to calculate `next_cache_start` here.
            new_cache = torch.cat((k, v), dim=-1)
        elif limited_context_attn:
            # chunking query
            # [B, time1, head, d_k]
            q_size = q.size(1)
            n_frames_pad = (chunk_size - ((q_size - chunk_size) % chunk_size)) 
            n_frames_pad = n_frames_pad % chunk_size
            q = torch.nn.functional.pad(q, (0, 0, 0, 0, 0, n_frames_pad))
            # [B, n_chunks, head, d_k, q_size]
            q = q.unfold(1, size=chunk_size, step=chunk_size)
            # [B * n_chunks, head, d_k, q_size]
            q = q.reshape(-1, q.size(2), q.size(3), q.size(4))
            # [B * n_chunks,q_size, head, d_k]
            q = q.permute(0, 3, 1, 2)

            # Chunking key and value
            # (batch, head, time1, d_k * 2)
            kv = torch.cat([k, v], dim=-1)
            kv = torch.nn.functional.pad(
                kv, 
                (0, 0, left_context_size, n_frames_pad + right_context_size))
            # [B, head, n_chunks, d_k * 2, l + c + r]
            kv = kv.unfold(
                2, 
                size=left_context_size + chunk_size + right_context_size, 
                step=chunk_size)
            # [B, n_chunks, head, l + c + r, d_k * 2]
            kv = kv.permute(0, 2, 1, 4, 3)
            # [B * n_chunks, head, l + c + r, d_k * 2]
            kv = kv.reshape(-1, kv.size(2), kv.size(3), kv.size(4))
            k, v = torch.split(kv, kv.size(-1) // 2, dim=-1)

            # Chunking mask for query
            # [B, 1, T + n_frames_pad]
            mask_q = torch.nn.functional.pad(mask, (0, n_frames_pad))
            # [B, 1, n_chunks, chunk_size]
            mask_q = mask_q.unfold(-1, size=chunk_size, step=chunk_size)
            # [B *n_chunks, chunk_size]
            mask_q = mask_q.reshape(-1, mask_q.size(-1))

            # Chunking mask for key and value
            mask_kv = torch.nn.functional.pad(
                mask, 
                (left_context_size, n_frames_pad + right_context_size))
            # [B, 1, n_chunks, chunk_size]
            mask_kv = mask_kv.unfold(
                -1, 
                size=left_context_size + chunk_size + right_context_size, 
                step=chunk_size)
            # [B, * n_chunks, chunk_size]
            mask_kv = mask_kv.reshape(-1, mask_kv.size(3))

            # finalize mask
            mask = mask_q.unsqueeze(-1) & mask_kv.unsqueeze(1)

            # return dummy new cache
            new_cache = cache
        else:
            new_cache = cache

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, time1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # (batch, head, time1, time2)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        # Add relative shift with left and right context inclusion, it can stream
        matrix_bd = self.rel_shift(matrix_bd, left_context_size, right_context_size)

        scores = (matrix_ac + matrix_bd) / math.sqrt(
            self.d_k)  # (batch, head, time1, time2)

        attn_output = self.forward_attention(v, scores, mask)
        if limited_context_attn:
            attn_output = attn_output.reshape(bz, -1, n_feat)
            attn_output = attn_output[:, :q_size, :]

        return attn_output, new_cache

    def forward_parallel_chunk(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        pos_emb: torch.Tensor = torch.empty(0),
        cache: torch.Tensor = torch.zeros((0, 0, 0)),
        right_context_size: int = 0,
        left_context_size: int = 0,
        truncated_context_size: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2), (0, 0, 0) means fake mask.
            pos_emb (torch.Tensor): Positional embedding tensor
                (#batch, time2, size).
            cache (torch.Tensor): Cache tensor (cache_t, head, d_k * 2),
                where `cache_t == left_context_size`
                and `head * d_k == size`
        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
            torch.Tensor: Cache tensor (cache_t, head, d_k * 2)
                where `cache_t == left_context_size`
                and `head * d_k == size`
        """
        q, k, v = self.forward_qkv(query, key, value)

        q = q.transpose(1, 2)  # (batch, time1, head, d_k)
        cache_t = cache.size(0)
        if cache_t == 0:
            cache = torch.zeros(
                (left_context_size, self.h, self.d_k * 2),
                device=q.device, dtype=q.dtype
            )
        # (B, head, time1, d_k * 2),
        kv = torch.cat([k, v], dim=-1)
        # [n_chunk * chunk_size, head, F]
        kv = kv.transpose(1, 2).reshape(-1, self.h, self.d_k * 2)


        # ----------Overlapping Chunk Transformation-----------------------------------
        kv = torch.cat([cache, kv], dim=0)

        if cache_t > 0:
            new_cache = kv[:truncated_context_size + cache.size(0)][-cache.size(0):]
        else:
            # Streaming long-form transcription is disabled if input cache is empty,
            new_cache = torch.zeros((0, 0, 0), device=q.device, dtype=q.dtype)
        kv = torch.nn.functional.pad(kv, (0, 0, 0, 0, 0, right_context_size))
        kv = kv.unfold(
            0,
            left_context_size + q.shape[1] + right_context_size,
            q.shape[1]
        )
        # -----------------------------------------------------------------------------

        # [n_chunk + 1, head, F, left_context_size]
        kv = kv.transpose(2, 3)
        k, v = torch.split(
            kv, kv.size(-1) // 2, dim=-1)

        # NOTE(xcsong): We do cache slicing in encoder.forward_chunk, since it's
        #   non-trivial to calculate `next_cache_start` here.
        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, time1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)


        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))


        # compute matrix b and matrix d
        # (batch, head, time1, time2)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))

        # Add relative shift with left and right context inclusion, it can stream
        matrix_bd = self.rel_shift(matrix_bd, left_context_size, right_context_size)

        scores = (matrix_ac + matrix_bd) / math.sqrt(
            self.d_k)  # (batch, head, time1, time2)

        return self.forward_attention(v, scores, mask), new_cache
