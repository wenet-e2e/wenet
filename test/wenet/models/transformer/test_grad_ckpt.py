#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright [2023-11-30] <sxc19@mails.tsinghua.edu.cn, Xingchen Song>

import torch
import pytest
import numpy as np

from wenet.transformer.encoder import TransformerEncoder, ConformerEncoder
from wenet.transformer.decoder import TransformerDecoder, BiTransformerDecoder


@pytest.mark.parametrize("module", [
    TransformerEncoder, ConformerEncoder, TransformerDecoder,
    BiTransformerDecoder
])
def test_encoder_gradient_checkpointing(module):
    torch.manual_seed(777)
    # Init model
    model = module(80,
                   256,
                   dropout_rate=0.0,
                   positional_dropout_rate=0.0,
                   gradient_checkpointing=False)
    model_grad_ckpt = module(80,
                             256,
                             dropout_rate=0.0,
                             positional_dropout_rate=0.0,
                             gradient_checkpointing=True)
    model_grad_ckpt.load_state_dict(model.state_dict(), strict=True)
    model.train()
    model_grad_ckpt.train()
    # Forward
    xs = torch.randn(2, 10, 80) + 10.0
    xs_lens = torch.tensor([10, 10], dtype=torch.long)
    tgt = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=torch.long)
    r_tgt = torch.tensor([[5, 4, 3, 2, 1], [10, 9, 8, 7, 6]], dtype=torch.long)
    tgt_lens = torch.tensor([5, 5], dtype=torch.long)
    memory = torch.randn(2, 10, 256) + 10.0
    memory_mask = torch.ones((2, 5, 10))
    if module in [TransformerEncoder, ConformerEncoder]:
        logits = model(xs, xs_lens)[0]
        logits_grad_ckpt = model_grad_ckpt(xs, xs_lens)[0]
    elif module in [TransformerDecoder, BiTransformerDecoder]:
        logits = model(memory, memory_mask, tgt, tgt_lens, r_tgt)[0]
        logits_grad_ckpt = model_grad_ckpt(memory, memory_mask, tgt, tgt_lens,
                                           r_tgt)[0]
    else:
        raise NotImplementedError
    np.testing.assert_allclose(logits.detach().numpy(),
                               logits_grad_ckpt.detach().numpy(),
                               rtol=1e-7,
                               atol=1e-10)
    # Backward
    model.zero_grad()
    logits.sum().backward()
    model_grad_ckpt.zero_grad()
    logits_grad_ckpt.sum().backward()
    for (name1, param1), (name2,
                          param2) in zip(model.named_parameters(),
                                         model_grad_ckpt.named_parameters()):
        assert name1 == name2
        if param1.grad is None or param2.grad is None:
            print("Ignore {}, due to grad = None".format(name1))
        elif not param1.requires_grad or not param2.requires_grad:
            print("Ignore {}, due to requires_grad = False".format(name1))
        else:
            np.testing.assert_allclose(param1.grad.detach().numpy(),
                                       param2.grad.detach().numpy(),
                                       rtol=1e-7,
                                       atol=1e-10)
            print("Pass {}".format(name1))
