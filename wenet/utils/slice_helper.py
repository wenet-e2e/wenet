# Copyright 2021 Huya Inc. All Rights Reserved.
# Author: lizexuan@huya.com (Zexuan Li)
# Reference from https://github.com/fanlu/wenet/blob/40062b065405280b5ae679c8e6d91a2333294d0a/wenet/transformer/slice_helper.py

import torch

@torch.jit.script
def slice_helper(x, offset):
    return x[:, -offset: , :]

@torch.jit.script
def slice_helper2(x: torch.Tensor, start, end):
    return x[:, start:end]

@torch.jit.script
def slice_helper3(x, start):
    return x[:, start:]

@torch.jit.script
def get_next_start(xs, required_cache_size):
    next_cache_start = 0
    if required_cache_size < 0:
        next_cache_start = 1
    elif required_cache_size == 0:
        next_cache_start = xs.size(1)
    else:
        if xs.size(1) - required_cache_size < 0:
            next_cache_start = 1
        else:
            next_cache_start = xs.size(1) - required_cache_size
    return torch.tensor(next_cache_start, dtype=torch.int64)
