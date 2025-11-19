import sys

import torch
import torch.nn as nn

# 1. Safe Import
# We wrap the import in a try-except block.
# If the package is missing, we set a flag instead of crashing.
try:
    from vector_quantize_pytorch import FSQ
    IS_FSQ_AVAILABLE = True
except ImportError:
    FSQ = None
    IS_FSQ_AVAILABLE = False


def _inject_single_layer(model, layer_idx, fsq_kwargs):
    """
    Internal function: Injects FSQ into a specific encoder layer.
    """
    # Double check availability inside the worker function just in case
    if not IS_FSQ_AVAILABLE:
        return

    try:
        # WeNet encoder layers are typically stored in 'model.encoder.encoders'
        target_layer = model.encoder.encoders[layer_idx]
    except IndexError:
        print(
            f"[FSQ Warning] Layer index {layer_idx} is out of bounds. Skipped."
        )
        return
    except AttributeError:
        print(
            f"[FSQ Error] Could not find 'model.encoder.encoders'. Check WeNet version."
        )
        return

    # Automatically retrieve the encoder output dimension.
    dim = getattr(model.encoder, 'output_size', lambda: 256)()

    # Initialize the FSQ module
    fsq_module = FSQ(dim=dim, **fsq_kwargs)

    # Assign module to layer (for state_dict preservation)
    target_layer.fsq_injected_module = fsq_module

    # Define the Hook
    def hook(module, input, output):
        #  (x, masks, pos_emb, mask_pad)
        if isinstance(output, tuple):
            x = output[0]
            masks = output[-1]  # [B, 1, T]
            others = output[1:-1]
        else:
            raise NotImplementedError

        quantized_x, indices = module.fsq_injected_module(x)
        if masks is not None:
            mask_for_mul = masks.transpose(1, 2)
            if mask_for_mul.dtype == torch.bool:
                mask_for_mul = mask_for_mul.to(quantized_x.dtype)
            quantized_x = quantized_x * mask_for_mul

        if others:
            return (quantized_x, masks) + others
        return quantized_x

    # Register the hook
    target_layer.register_forward_hook(hook)
    print(
        f"[FSQ Info] Injected FSQ after Encoder Layer {layer_idx} | Params: {fsq_kwargs}"
    )


def apply_fsq_configuration(model, configs):
    """
    Main Entry Point: Parses WeNet configs and applies FSQ injection.
    Safe to call even if vector_quantize_pytorch is not installed.
    """
    fsq_conf = configs.get('fsq_conf', None)

    # 1. Check if FSQ is enabled in YAML
    if not fsq_conf or not fsq_conf.get('enable', False):
        return model

    # 2. Check if the library is actually installed
    if not IS_FSQ_AVAILABLE:
        print("[FSQ Warning] FSQ is enabled in config,",
              "but 'vector_quantize_pytorch' is not installed.")
        print("[FSQ Warning] FSQ injection will be SKIPPED.")
        print("[FSQ Warning] To fix: pip install vector-quantize-pytorch")
        return model

    # 3. Parse configuration and inject
    indices = fsq_conf.get('insert_indices', [])
    fsq_args = fsq_conf.get('fsq_args', {})

    if isinstance(indices, int):
        indices = [indices]

    for idx in indices:
        _inject_single_layer(model, idx, fsq_args)

    return model
