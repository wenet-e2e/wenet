# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import print_function

import argparse
import logging
import os

import torch
import yaml

from wenet.utils.init_model import init_model
import intel_extension_for_pytorch as ipex
from intel_extension_for_pytorch.quantization import prepare, convert


def get_args():
    parser = argparse.ArgumentParser(description='export your script model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--output_file', default=None, help='output file')
    parser.add_argument('--dtype',
                        default="fp32",
                        help='choose the dtype to run:[fp32,bf16]')
    parser.add_argument('--output_quant_file',
                        default=None,
                        help='output quantized model file')
    args = parser.parse_args()
    return args


def scripting(model):
    with torch.inference_mode():
        script_model = torch.jit.script(model)
        script_model = torch.jit.freeze(
            script_model,
            preserved_attrs=[
                "forward_encoder_chunk", "ctc_activation",
                "forward_attention_decoder", "subsampling_rate",
                "right_context", "sos_symbol", "eos_symbol",
                "is_bidirectional_decoder"
            ])
    return script_model


def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    # No need gpu for model export
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    model, configs = init_model(args, configs)
    print(model)

    # Apply IPEX optimization
    model.eval()
    torch._C._jit_set_texpr_fuser_enabled(False)
    model.to(memory_format=torch.channels_last)
    if args.dtype == "fp32":
        ipex_model = ipex.optimize(model)
    elif args.dtype == "bf16":  # For Intel 4th generation Xeon (SPR)
        ipex_model = ipex.optimize(model,
                                   dtype=torch.bfloat16,
                                   weights_prepack=False)

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
        dynamic_qconfig = ipex.quantization.default_dynamic_qconfig
        dummy_data = (torch.zeros(1, 67, 80), 16, -16,
                      torch.zeros(12, 4, 32, 128), torch.zeros(12, 1, 256, 7))
        model = prepare(model, dynamic_qconfig, dummy_data)
        model = convert(model)
        script_quant_model = scripting(model)
        script_quant_model.save(args.output_quant_file)
        print('Export quantized model successfully, '
              'see {}'.format(args.output_quant_file))


if __name__ == '__main__':
    main()
