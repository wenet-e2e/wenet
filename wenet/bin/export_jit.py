# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Di Wu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import argparse
import os

import torch
import yaml

from wenet.utils.checkpoint import load_checkpoint
from wenet.utils.init_model import init_model


def get_args():
    parser = argparse.ArgumentParser(description='export your script model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--output_file', default=None, help='output file')
    parser.add_argument('--output_quant_file',
                        default=None,
                        help='output quantized model file')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    # No need gpu for model export
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    model = init_model(configs)
    print(model)

    load_checkpoint(model, args.checkpoint)
    # Export jit torch script model

    if args.output_file:
        script_model = torch.jit.script(model)
        script_model.save(args.output_file)
        print('Export model successfully, see {}'.format(args.output_file))

    # Export quantized jit torch script model
    if args.output_quant_file:
        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        print(quantized_model)
        script_quant_model = torch.jit.script(quantized_model)
        script_quant_model.save(args.output_quant_file)
        print('Export quantized model successfully, '
              'see {}'.format(args.output_quant_file))


if __name__ == '__main__':
    main()
