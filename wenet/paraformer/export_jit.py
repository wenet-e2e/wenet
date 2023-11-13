""" NOTE(Mddct): This file is experimental and is used to export paraformer
"""

import argparse
import logging
import torch
import yaml

from wenet.utils.init_ali_paraformer import init_model


def get_args():
    parser = argparse.ArgumentParser(description='load ali-paraformer')
    parser.add_argument('--ali_paraformer',
                        required=True,
                        help='ali released Paraformer model path')
    parser.add_argument('--config', required=True, help='config of paraformer')
    parser.add_argument('--output_file', default=None, help='output file')
    args = parser.parse_args()
    return args


def main():

    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    model, _ = init_model(configs, args.ali_paraformer)
    model.eval()

    if args.output_file:
        script_model = torch.jit.script(model)
        script_model.save(args.output_file)


if __name__ == "__main__":

    main()
