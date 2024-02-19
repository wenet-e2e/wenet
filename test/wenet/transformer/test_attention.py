#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright [2023-11-30] <sxc19@mails.tsinghua.edu.cn, Xingchen Song>

import torch
import pytest
from wenet.transformer.attention import MultiHeadedAttention

from wenet.utils.mask import add_optional_chunk_mask, make_non_pad_mask


@pytest.mark.parametrize("args", [
    {
        "n_feat": 256,
        "n_head": 4,
        "dropout_rate": 0.0
    },
    {
        "n_feat": 512,
        "n_head": 8,
        "dropout_rate": 0.0
    },
    {
        "n_feat": 1280,
        "n_head": 20,
        "dropout_rate": 0.0
    },
    {
        "n_feat": 512,
        "n_head": 4,
        "dropout_rate": 0.0
    },
])
def test_sdpa(args):
    torch.manual_seed(777)
    mha_module = MultiHeadedAttention(use_sdpa=False, **args)
    torch.manual_seed(777)
    mha_module_with_sdpa = MultiHeadedAttention(use_sdpa=True, **args)
    mha_module.eval()
    mha_module_with_sdpa.eval()

    q = torch.rand(10, 100, args['n_feat'])
    k = torch.rand(10, 100, args['n_feat'])
    v = torch.rand(10, 100, args['n_feat'])
    input_lens = torch.tensor([100, 90, 80, 79, 60, 51, 40, 30, 10, 5])
    mask = make_non_pad_mask(input_lens).unsqueeze(1)
    att_mask = add_optional_chunk_mask(q,
                                       mask,
                                       use_dynamic_chunk=True,
                                       decoding_chunk_size=0,
                                       static_chunk_size=0,
                                       use_dynamic_left_chunk=True,
                                       num_decoding_left_chunks=-1)
    output, cache = mha_module(q, k, v, mask=att_mask)

    att_mask = (1.0 - att_mask.float()) * torch.finfo(torch.float).min
    output_with_sdpa, cache_with_sdpa = mha_module_with_sdpa(q,
                                                             k,
                                                             v,
                                                             mask=att_mask)

    assert torch.allclose(
        output * mask.transpose(1, 2),
        output_with_sdpa * mask.transpose(1, 2),
        atol=9e-7,
        rtol=9e-4,
    )
    assert torch.allclose(cache, cache_with_sdpa)
