# NOTE(Mddct): This file is to convert paraformer config to wenet's train.yaml config

import argparse
import json
import math
import os
from typing import List, Tuple

import yaml


def _load_paraformer_cmvn(cmvn_file) -> Tuple[List, List]:
    with open(cmvn_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    means_list = []
    vars_list = []
    for i in range(len(lines)):
        line_item = lines[i].split()
        if line_item[0] == '<AddShift>':
            line_item = lines[i + 1].split()
            if line_item[0] == '<LearnRateCoef>':
                add_shift_line = line_item[3:(len(line_item) - 1)]
                means_list = list(map(float, list(add_shift_line)))
                continue
        elif line_item[0] == '<Rescale>':
            line_item = lines[i + 1].split()
            if line_item[0] == '<LearnRateCoef>':
                rescale_line = line_item[3:(len(line_item) - 1)]
                vars_list = list(map(float, list(rescale_line)))
                continue

    for i in range(len(means_list)):
        # paraformer mean is negative
        means_list[i] = -means_list[i]
        vars_list[i] = 1. / math.pow(vars_list[i],
                                     2) + means_list[i] * means_list[i]
    return means_list, vars_list


def _filter_dict_fields(input_dict, fields_to_keep):
    filtered_dict = {
        key: value
        for key, value in input_dict.items() if key in fields_to_keep
    }
    return filtered_dict


def _to_wenet_cmvn(cmvn_file):
    means, istd = _load_paraformer_cmvn(cmvn_file)

    d = {}
    d['mean_stat'] = means
    d['var_stat'] = istd
    d['frame_num'] = 1

    return json.dumps(d)


def extract_dict(configs, wenet_dict_path: str) -> int:
    tokens = configs['token_list']
    with open(wenet_dict_path, '+w') as f:
        for i, token in enumerate(tokens):
            token = '<sos>' if token == '<s>' else token
            token = '<eos>' if token == '</s>' else token
            f.writelines(token + ' ' + str(i) + '\n')

        f.flush()
    return len(tokens)


def convert_to_wenet_json_cmvn(paraformer_cmvn_path, wenet_cmvn_path: str):
    json_cmvn = _to_wenet_cmvn(paraformer_cmvn_path)
    with open(wenet_cmvn_path, '+w') as f:
        f.write(json_cmvn)
        f.flush()


def convert_to_wenet_yaml(configs, wenet_yaml_path: str, fields_to_keep: List):
    configs = _filter_dict_fields(configs, fields_to_keep)
    configs['encoder'] = 'SanmEncoder'
    configs['encoder_conf']['input_layer'] = 'conv2d'
    configs['decoder'] = 'SanmDecoder'
    configs['lfr_conf'] = {'lfr_m': 7, 'lfr_n': 6}

    configs['cif_predictor_conf'] = configs.pop('predictor_conf')
    configs['cif_predictor_conf']['cnn_groups'] = 1
    configs['cif_predictor_conf']['residual'] = False
    # This type not use
    del configs['encoder_conf']['selfattention_layer_type'], configs[
        'encoder_conf']['pos_enc_class']

    configs['dataset_conf'] = {}
    configs['dataset_conf']['filte_conf'] = {}
    configs['dataset_conf']['speed_perturn'] = False
    configs['dataset_conf']['spec_aug'] = False
    configs['dataset_conf']['spec_sub'] = False
    configs['dataset_conf']['spec_trim'] = False
    configs['dataset_conf']['shuffle'] = False
    configs['dataset_conf']['sort'] = False

    configs['grad_clip'] = 5
    configs['accum_grad'] = 1
    configs['max_epoch'] = 100
    configs['log_interval'] = 100

    with open(wenet_yaml_path, '+w') as f:
        f.write(yaml.dump(configs))
        f.flush()


def get_args():
    parser = argparse.ArgumentParser(description='load ali-paraformer')
    parser.add_argument('--paraformer_config',
                        required=True,
                        help='ali released Paraformer model\'s config')
    parser.add_argument('--paraformer_cmvn',
                        required=True,
                        help='ali released Paraformer model\'s cmvn')
    parser.add_argument(
        '--output_dir',
        default='.',
        help='output file in wenet\'s style: global_cmvn, units.txt, train.yaml'
    )
    args = parser.parse_args()
    return args


def main():

    args = get_args()
    assert os.path.exists(args.output_dir)
    with open(args.paraformer_config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    json_cmvn_path = os.path.join(args.output_dir, 'global_cmvn')
    convert_to_wenet_json_cmvn(args.paraformer_cmvn, json_cmvn_path)
    vocab_size = extract_dict(configs,
                              os.path.join(args.output_dir, 'units.txt'))
    configs['paraformer'] = True
    configs['is_json_cmvn'] = True
    configs['cmvn_file'] = json_cmvn_path
    configs['input_dim'] = 80
    configs['output_dim'] = vocab_size
    fields_to_keep = [
        'encoder_conf', 'decoder_conf', 'predictor_conf', 'input_dim',
        'output_dim', 'cmvn_file', 'is_json_cmvn', 'model_conf', 'paraformer',
        'optim', 'optim_conf', 'scheduler', 'scheduler_conf'
    ]
    convert_to_wenet_yaml(configs, os.path.join(args.output_dir, 'train.yaml'),
                          fields_to_keep)


if __name__ == "__main__":

    main()
