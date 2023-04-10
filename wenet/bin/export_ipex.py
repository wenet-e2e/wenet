# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import print_function

import argparse
import os

import torch
import yaml

from wenet.utils.checkpoint import load_checkpoint
from wenet.utils.init_model import init_model
import intel_extension_for_pytorch as ipex

def get_args():
    parser = argparse.ArgumentParser(description='export your script model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--output_file', default=None, help='output file')
    parser.add_argument('--dtype', default="fp32", help='choose the dtype to run:[fp32,bf16]')
    parser.add_argument('--output_quant_file',
                        default=None,
                        help='output quantized model file')
    args = parser.parse_args()
    return args

def scripting(model):
    script_model = torch.jit.script(model)
    script_model = torch.jit.freeze(
                    script_model,
                    preserved_attrs=["forward_encoder_chunk",
                                     "ctc_activation",
                                     "forward_attention_decoder",
                                     "subsampling_rate",
                                     "right_context",
                                     "sos_symbol",
                                     "eos_symbol",
                                     "is_bidirectional_decoder"]
                    )
    return script_model

def main():
    args = get_args()
    # No need gpu for model export
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    model = init_model(configs)
    print(model)

    load_checkpoint(model, args.checkpoint)

    # Apply IPEX optimization
    model.eval()
    torch._C._jit_set_texpr_fuser_enabled(False)
    model.to(memory_format=torch.channels_last)
    if args.dtype == "fp32":
        ipex_model = ipex.optimize(model)
    elif args.dtype == "bf16": # For Intel 4th generation Xeon (SPR)
        ipex_model = ipex.optimize(model, dtype=torch.bfloat16, weights_prepack=False)

    # Export jit torch script model
    if args.output_file:
        if args.dtype == "fp32":
            script_model = scripting(ipex_model)
        elif args.dtype == "bf16":
            torch._C._jit_set_autocast_mode(True)
            with torch.cpu.amp.autocast():
                script_model = scripting(ipex_model)
        script_model.save(args.output_file)
        print('Export model successfully, see {}'.format(args.output_file))

    # Export quantized jit torch script model
    if args.output_quant_file:
        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        print(quantized_model)
        script_quant_model = scripting(quantized_model)
        script_quant_model.save(args.output_quant_file)
        print('Export quantized model successfully, '
              'see {}'.format(args.output_quant_file))


if __name__ == '__main__':
    main()
