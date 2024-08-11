# Copyright (c) 2021 microsoft
#               2023 Alan (alanfangemail@gmail.com)
#  -----------------------------------------------------------------------------
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for
#  license information.
#  -----------------------------------------------------------------------------

import logging
import torch
import torch.nn as nn

from typing import Dict, List

import wenet.finetune.lora.layers as lora


def get_nested_attr(module, attr_path):
    attrs = attr_path.split('.')
    for attr in attrs:
        if hasattr(module, attr):
            module = getattr(module, attr)
        else:
            return None
    return module


def inject_lora(module, lora_config):
    lora_rank = lora_config["lora_rank"]
    lora_alpha = lora_config["lora_alpha"]
    lora_dropout = lora_config["lora_dropout"]
    for lora_attr in lora_config["lora_list"]:
        if hasattr(module, lora_attr):
            submodule = getattr(module, lora_attr)
            n_feat = submodule.in_features
            lora_linear = lora.Linear(n_feat, n_feat, r=lora_rank,
                                      lora_alpha=lora_alpha,
                                      lora_dropout=lora_dropout)
            setattr(module, lora_attr, lora_linear)


def inject_lora_to_model(model, lora_config):
    lora_modules = []
    for module in lora_config["lora_modules"]:
        submodule = get_nested_attr(model, module)
        for layer in submodule:
            lora_modules.append(layer)

    updated_lora_modules = []
    for i in range(len(lora_modules)):
        for attn_attr in lora_config["lora_attn_attr"]:
            if hasattr(lora_modules[i], attn_attr):
                updated_lora_modules.append(getattr(lora_modules[i], attn_attr))

    for lora_module in updated_lora_modules:
        inject_lora(lora_module, lora_config)


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
            if isinstance(m, lora.LoRALayer) and \
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


def get_record_gradient_hook(model, record_dict):
    def record_gradient_hook(grad):
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                if n not in record_dict:
                    record_dict[n] = p.grad.cpu()
                else:
                    record_dict[n] += p.grad.cpu()
                p.grad = None
        return grad

    return record_gradient_hook


def estimate_gradient(
    model, dataloader, max_iters: int = 8,
    device: torch.device = torch.device("cpu")
) -> Dict[str, List[torch.Tensor]]:
    r"""
    Estimate the gradient of the model on the given dataset
    """
    logging.info("Estimating gradient layer by layer, time needed")
    model.train()
    named_grads = {}
    hooks = []
    requires_grad_states = {}
    for name, param in model.named_parameters():
        requires_grad_states[name] = param.requires_grad
        param.requires_grad = True
        hook = param.register_hook(get_record_gradient_hook(model, named_grads))
        hooks.append(hook)
    num = 0
    for _, batch_dict in enumerate(dataloader):
        num += 1
        if max_iters is not None and num >= max_iters:
            break
        outputs = model(batch_dict, device)
        outputs['loss'].backward()
        get_record_gradient_hook(model, named_grads)(None)  # get gradient of last layer
        # make sure the gradient is cleared
        for n, p in model.named_parameters():
            if p.grad is not None:
                p.grad = None
    for n, _ in named_grads.items():
        named_grads[n] /= num
    for hook in hooks:
        hook.remove()
    # recover original requires_grad states
    for name, param in model.named_parameters():
        param.requires_grad = requires_grad_states[name]
    torch.cuda.empty_cache()
    return named_grads


@torch.no_grad()
def reinit_lora_modules(name, module, init_config, **kwargs):
    r"""Refer to https://github.com/Outsider565/LoRA-GA/blob/
    c185846309ea9012d0bcd46ebd30347dda1c592c/run_exp.py#L67
    Reinitialize the lora model with the given configuration.
    """
    import math
    lora_r = min(module.lora_A.shape)
    a_dim = max(module.lora_A.shape)
    b_dim = max(module.lora_B.shape)
    if init_config.mode == "simple":
        match init_config.lora_A:
            case "gaussian":
                torch.nn.init.normal_(
                    module.lora_A, mean=0.0,
                    std=init_config.lora_A_std
                )
            case "kaiming":
                # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
                torch.nn.init.kaiming_uniform_(module.lora_A,
                                               a=math.sqrt(5))
            case "fan_out_kaiming":
                torch.nn.init.kaiming_normal_(
                    module.lora_A, mode="fan_out"
                )
            case "xavier":
                torch.nn.init.xavier_normal_(module.lora_A)
            case "zeros":
                torch.nn.init.zeros_(module.lora_A)
            case "unit":
                torch.nn.init.normal_(
                    module.lora_A, mean=0.0,
                    std=1.0 / (a_dim**0.5)
                )
            case "orthogonal":
                torch.nn.init.orthogonal_(module.lora_A)
            case _:
                raise ValueError(
                    f"Unknown lora_A initialization: {init_config.lora_A}"
                )
        match init_config.lora_B:
            case "gaussian":
                torch.nn.init.normal_(
                    module.lora_B, mean=0.0,
                    std=init_config.lora_B_std
                )
            case "kaiming":
                torch.nn.init.kaiming_normal_(module.lora_B)
            case "fan_out_kaiming":
                torch.nn.init.kaiming_normal_(
                    module.lora_B, mode="fan_out"
                )
            case "xavier":
                torch.nn.init.xavier_normal_(module.lora_B)
            case "zeros":
                torch.nn.init.zeros_(module.lora_B)
            case "unit":
                torch.nn.init.normal_(
                    module.lora_B, mean=0.0,
                    std=1.0 / (b_dim**0.5)
                )
            case "orthogonal":
                torch.nn.init.orthogonal_(module.lora_B)
            case _:
                raise ValueError(
                    f"Unknown lora_B initialization: {init_config.lora_B}"
                )
        if getattr(init_config, 'scale', '') == "stable":
            gamma = init_config.stable_gamma
            m, n = module.weight.shape
            module.lora_B.data *= (m**0.25) / gamma**0.5
            module.lora_A.data *= (n**0.25) / gamma**0.5
    elif init_config.mode == "svd":
        U, S, V = torch.svd_lowrank(module.weight.float(), q=4 * lora_r,
                                    niter=4)
        V = V.T
        m, n = module.weight.shape
        if init_config.scale == "default":
            S = S / module.scaling
            module.lora_B = torch.nn.Parameter(
                (U[:, :lora_r] * torch.sqrt(S[:lora_r])).contiguous()
            )
            module.lora_A = torch.nn.Parameter(
                (V[:lora_r, :].T * torch.sqrt(S[:lora_r])).T.contiguous()
            )
        elif init_config.scale == "stable":
            gamma = init_config.stable_gamma
            module.lora_B = torch.nn.Parameter(
                (U[:, :lora_r] * (m**0.25) / gamma**0.5).contiguous()
            )
            module.lora_A = torch.nn.Parameter(
                (V[:lora_r, :] * (n**0.25) / gamma**0.5).contiguous()
            )
        elif init_config.scale == "unit":
            module.lora_B = torch.nn.Parameter((U[:, :lora_r]).contiguous())
            module.lora_A = torch.nn.Parameter((V[:lora_r, :]).contiguous())
        elif init_config.scale == "normalized":
            S_sum = S[:lora_r].sum()
            module.lora_B = torch.nn.Parameter(
                (U[:, :lora_r] * torch.sqrt(S[:lora_r])
                 / torch.sqrt(S_sum) * lora_r**0.5).contiguous()
            )
            module.lora_A = torch.nn.Parameter(
                (V[:lora_r, :].T * torch.sqrt(S[:lora_r])
                 / torch.sqrt(S_sum) * lora_r**0.5).T.contiguous()
            )
    elif init_config.mode == "gradient":
        named_grad = kwargs["named_grads"]
        grad_name = name + ".weight"
        grads = named_grad[grad_name]
        U, S, V = torch.svd_lowrank(grads.cuda().float(), q=4 * lora_r, niter=4)
        V = V.T
        # set direction
        if init_config.direction == "ArBr":
            B = U[:, 0 : 2 * lora_r : 2]
            A = V[1 : 2 * lora_r : 2, :]
        elif init_config.direction == "A2rBr":
            B = U[:, :lora_r]
            A = V[lora_r : 2 * lora_r, :]
        elif init_config.direction == "ArB2r":
            B = U[:, lora_r : 2 * lora_r]
            A = V[:lora_r, :]
        scaling_factor = module.scaling
        if init_config.scale == "gd":
            A = A / scaling_factor
            B = B / scaling_factor
        elif init_config.scale == "unit":
            # Because A,B is orthogonal, do not need to scale
            pass
        elif init_config.scale == "stable":
            m, n = grads.shape
            # m: feature_out, n: feature_in
            # the scale of output is only related to the feature_out
            gamma = init_config.stable_gamma
            B = B * m**0.25 / gamma**0.5
            A = A * m**0.25 / gamma**0.5
        elif init_config.scale == "weightS":
            _, S, _ = torch.svd_lowrank(module.weight.float(), q=4 * lora_r,
                                        niter=4)
            S = S / module.scaling
            avg_s = torch.sqrt(S[:lora_r]).mean().to(A.device)
            B = B * avg_s
            A = A * avg_s
        module.lora_B = torch.nn.Parameter(B.contiguous().cuda())
        module.lora_A = torch.nn.Parameter(A.contiguous().cuda())

    with torch.no_grad():
        # consider dtype not in init_config
        if not hasattr(init_config, "dtype"):
            pass
        elif init_config.dtype == "bf16":
            module.lora_A.data = module.lora_A.data.to(torch.bfloat16)
            module.lora_B.data = module.lora_B.data.to(torch.bfloat16)
        elif init_config.dtype == "fp32":
            module.lora_A.data = module.lora_A.data.to(torch.float32)
            module.lora_B.data = module.lora_B.data.to(torch.float32)
        # If lora_A@lora_B is not zero,
        # then we need to subtract lora_A@lora_B from the original weight matrix
        offset = (
            module.lora_B @ module.lora_A
        ).to(module.weight.data.device)
        scaling_factor = module.scaling
        offset *= scaling_factor
        if hasattr(init_config, "norm_clip") and init_config.norm_clip:
            # for numerical stability,
            # offset's largest value must be less then weight's largest value
            ratio = torch.max(torch.abs(module.weight.data)) / torch.max(
                torch.abs(offset)
            )
            if ratio < 1:
                offset *= ratio
                module.lora_A.data *= ratio**0.5
                module.lora_B.data *= ratio**0.5
                logging.warning(f"Clipping offset by {ratio}")
        try:
            module.weight.data -= offset
        except Exception as e:
            logging.warning(f"{e}")
            breakpoint()
