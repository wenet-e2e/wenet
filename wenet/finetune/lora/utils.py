# Copyright (c) 2021 microsoft
#               2023 Alan (alanfangemail@gmail.com)
#  -----------------------------------------------------------------------------
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for
#  license information.
#  -----------------------------------------------------------------------------

import logging
import torch
import torch.nn as nn

from typing import Dict

from wenet.finetune.lora.attention import (LoRARelPositionMultiHeadedAttention,
                                           LoRAMultiHeadedAttention)
from wenet.finetune.lora.layers import LoRALayer

WENET_LORA_ATTENTION_CLASSES = {
    "selfattn": LoRAMultiHeadedAttention,
    "rel_selfattn": LoRARelPositionMultiHeadedAttention,
}


def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    logging.info('freezing all params except lora module.')
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRALayer) and \
               hasattr(m, 'bias') and \
               m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


def lora_state_dict(model: nn.Module,
                    bias: str = 'none') -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
    elif bias == 'all':
        return {
            k: my_state_dict[k]
            for k in my_state_dict if 'lora_' in k or 'bias' in k
        }
    elif bias == 'lora_only':
        to_return = {}
        for k in my_state_dict:
            if 'lora_' in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split('lora_')[0] + 'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError
