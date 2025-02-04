# NOTE(Mddct): This file is to convert paraformer config to wenet's train.yaml config

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import urllib.request
import torch
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

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


def convert_to_wenet_tokenizer_conf(symbol_table_path, seg_dict, configs,
                                    output_path):
    configs['tokenizer'] = 'paraformer'
    configs['tokenizer_conf'] = {}
    configs['tokenizer_conf']['symbol_table_path'] = symbol_table_path
    configs['tokenizer_conf']['seg_dict_path'] = output_path
    configs['tokenizer_conf']['special_tokens'] = {}
    configs['tokenizer_conf']['special_tokens']['<eos>'] = 2
    configs['tokenizer_conf']['special_tokens']['<sos>'] = 1
    configs['tokenizer_conf']['special_tokens']['<blank>'] = 0
    configs['tokenizer_conf']['special_tokens']['<unk>'] = 8403

    shutil.copy(seg_dict, output_path)


def convert_to_wenet_yaml(configs, wenet_yaml_path: str,
                          fields_to_keep: List[str]) -> Dict:
    configs = _filter_dict_fields(configs, fields_to_keep)
    configs['encoder'] = 'sanm_encoder'
    configs['encoder_conf']['input_layer'] = 'paraformer_dummy'
    configs['decoder'] = 'sanm_decoder'
    configs['lfr_conf'] = {'lfr_m': 7, 'lfr_n': 6}

    configs['input_dim'] = configs['lfr_conf']['lfr_m'] * 80
    # configs['predictor'] = 'cif_predictor'
    configs['predictor'] = 'paraformer_predictor'
    configs['predictor_conf'] = configs.pop('predictor_conf')
    configs['predictor_conf']['cnn_groups'] = 1
    configs['predictor_conf']['residual'] = False
    del configs['predictor_conf']['upsample_type']
    del configs['predictor_conf']['use_cif1_cnn']
    # This type not use
    del configs['encoder_conf']['selfattention_layer_type'], configs[
        'encoder_conf']['pos_enc_class']
    configs['encoder_conf']['pos_enc_layer_type'] = 'abs_pos_paraformer'

    configs['ctc_conf'] = {}
    configs['ctc_conf']['ctc_blank_id'] = 0

    configs['dataset_conf'] = {}
    configs['dataset_conf']['filter_conf'] = {}
    configs['dataset_conf']['filter_conf']['max_length'] = 20000
    configs['dataset_conf']['filter_conf']['min_length'] = 0
    configs['dataset_conf']['filter_conf']['token_max_length'] = 200
    configs['dataset_conf']['filter_conf']['token_min_length'] = 1
    configs['dataset_conf']['resample_conf'] = {}
    configs['dataset_conf']['resample_conf']['resample_rate'] = 16000
    configs['dataset_conf']['speed_perturb'] = True
    configs['dataset_conf']['spec_aug'] = True
    configs['dataset_conf']['spec_aug_conf'] = {}
    configs['dataset_conf']['spec_aug_conf']['num_t_mask'] = 2
    configs['dataset_conf']['spec_aug_conf']['num_f_mask'] = 2
    configs['dataset_conf']['spec_aug_conf']['max_t'] = 50
    configs['dataset_conf']['spec_aug_conf']['max_f'] = 10
    configs['dataset_conf']['fbank_conf'] = {}
    configs['dataset_conf']['fbank_conf']['num_mel_bins'] = 80
    configs['dataset_conf']['fbank_conf']['frame_shift'] = 10
    configs['dataset_conf']['fbank_conf']['frame_length'] = 25
    configs['dataset_conf']['fbank_conf']['dither'] = 0.1
    configs['dataset_conf']['fbank_conf']['window_type'] = 'hamming'
    configs['dataset_conf']['spec_sub'] = False
    configs['dataset_conf']['spec_trim'] = False
    configs['dataset_conf']['shuffle'] = True
    configs['dataset_conf']['shuffle_conf'] = {}
    configs['dataset_conf']['shuffle_conf']['shuffle_size'] = 1500
    configs['dataset_conf']['sort'] = True
    configs['dataset_conf']['sort_conf'] = {}
    configs['dataset_conf']['sort_conf']['sort_size'] = 500
    configs['dataset_conf']['batch_conf'] = {}
    configs['dataset_conf']['batch_conf']['batch_type'] = 'dynamic'
    configs['dataset_conf']['batch_conf']['batch_size'] = 26
    configs['dataset_conf']['batch_conf']['max_frames_in_batch'] = 12000

    configs['model_conf']['add_eos'] = configs['model_conf']['predictor_bias']
    del configs['model_conf']['predictor_bias']
    del configs['model_conf']['predictor_weight']

    configs['grad_clip'] = 5
    configs['accum_grad'] = 1
    configs['max_epoch'] = 100
    configs['log_interval'] = 100

    configs['model_conf']['length_normalized_loss'] = False

    with open(wenet_yaml_path, '+w') as f:
        f.write(yaml.dump(configs))
        f.flush()
    return configs


def convert_to_wenet_state_dict(args, wenet_model_path):
    wenet_state_dict = {}
    checkpoint = torch.load(args.paraformer_model, map_location='cpu')
    for name in checkpoint.keys():
        wenet_name = name

        if wenet_name.startswith('predictor.cif_output2'):
            wenet_name = wenet_name.replace('predictor.cif_output2.',
                                            'predictor.tp_output.')
        elif wenet_name.startswith('predictor.cif'):
            wenet_name = wenet_name.replace('predictor.cif',
                                            'predictor.predictor.cif')
        elif wenet_name.startswith('predictor.upsample'):
            wenet_name = wenet_name.replace('predictor.', 'predictor.tp_')
        elif wenet_name.startswith('predictor.blstm'):
            wenet_name = wenet_name.replace('predictor.', 'predictor.tp_')
        elif wenet_name == 'decoder.embed.0.weight':
            wenet_name = 'embed.weight'

        wenet_state_dict[wenet_name] = checkpoint[name].float()

    torch.save(wenet_state_dict, wenet_model_path)


def get_args():
    parser = argparse.ArgumentParser(description='load ali-paraformer')
    parser.add_argument('--paraformer_config',
                        default=None,
                        help='ali released Paraformer model\'s config')
    parser.add_argument('--paraformer_cmvn',
                        default=None,
                        help='ali released Paraformer model\'s cmvn')
    parser.add_argument('--paraformer_seg_dict',
                        default=None,
                        help='ali released Paraformer model\'s en dict')
    parser.add_argument('--output_dir',
                        default='.',
                        help="output file:\
        global_cmvn, units.txt, train.yaml, wenet_paraformer.pt")
    parser.add_argument("--paraformer_model",
                        default=None,
                        help="ali released Paraformer model")
    args = parser.parse_args()
    return args


def _download_fn(output_dir,
                 name,
                 renmae: Optional[str] = None,
                 version: str = 'master'):
    url = "https://www.modelscope.cn/api/v1/"\
        "models/iic/"\
        "speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"\
        "/repo?Revision={}&FilePath=".format(version) + name
    print(url)
    # "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"\
    if renmae is None:
        output_file = os.path.join(output_dir, name)
    else:
        output_file = os.path.join(output_dir, renmae)

    user_agent = "Mozilla/5.0"
    req = urllib.request.Request(url)
    req.add_header("User-Agent", user_agent)
    response = urllib.request.urlopen(req)
    file_size = int(response.headers["Content-Length"])

    with tqdm(total=file_size, unit='B', unit_scale=True, ncols=80,
              desc=name) as pbar:
        with urllib.request.urlopen(req) as response:
            with open(output_file, "wb") as file:
                while True:
                    data = response.read(4096)
                    if not data:
                        break
                    file.write(data)
                    pbar.update(len(data))
    print("{} download finished".format(name))


def may_get_assets_and_refine_args(args):

    assets_dir = os.path.join(Path.home(),
                              ".wenet/cache/paraformer-offline-cn")

    if not os.path.exists(assets_dir):
        os.makedirs(assets_dir)

    # TODO: md5 check
    if args.paraformer_config is None:
        config_name = 'config.yaml'
        args.paraformer_config = os.path.join(assets_dir, config_name)
        if not os.path.exists(args.paraformer_config):
            _download_fn(assets_dir, config_name, version='v1.2.4')
    if args.paraformer_cmvn is None:
        cmvn_name = 'am.mvn'
        args.paraformer_cmvn = os.path.join(assets_dir, cmvn_name)
        if not os.path.exists(args.paraformer_cmvn):
            _download_fn(assets_dir, cmvn_name)
    if args.paraformer_seg_dict is None:
        seg_dict = 'seg_dict'
        args.paraformer_seg_dict = os.path.join(assets_dir, "seg_dict")
        if not os.path.exists(args.paraformer_seg_dict):
            _download_fn(assets_dir, seg_dict)
    if args.paraformer_model is None:
        model_name = 'model.pt'
        args.paraformer_model = os.path.join(assets_dir, "model.pt")
        if not os.path.exists(args.paraformer_model):
            _download_fn(assets_dir, model_name, "model.pt")


def main():

    args = get_args()
    may_get_assets_and_refine_args(args)
    assert os.path.exists(args.output_dir)
    with open(args.paraformer_config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    json_cmvn_path = os.path.join(args.output_dir, 'global_cmvn')
    convert_to_wenet_json_cmvn(args.paraformer_cmvn, json_cmvn_path)

    wenet_units = os.path.join(args.output_dir, 'units.txt')
    seg_dict = os.path.join(args.output_dir,
                            os.path.basename(args.paraformer_seg_dict))
    vocab_size = extract_dict(configs, wenet_units)
    convert_to_wenet_tokenizer_conf(wenet_units, args.paraformer_seg_dict,
                                    configs, seg_dict)
    configs['output_dim'] = vocab_size
    configs['model'] = 'paraformer'
    configs['cmvn'] = "global_cmvn"
    configs['cmvn_conf'] = {}
    configs['cmvn_conf']['is_json_cmvn'] = True
    configs['cmvn_conf']['cmvn_file'] = json_cmvn_path
    fields_to_keep = [
        'model', 'encoder_conf', 'decoder_conf', 'predictor_conf', 'input_dim',
        'output_dim', 'cmvn', 'cmvn_conf', 'model_conf', 'paraformer', 'optim',
        'optim_conf', 'scheduler', 'scheduler_conf', 'tokenizer',
        'tokenizer_conf'
    ]
    wenet_train_yaml = os.path.join(args.output_dir, "train.yaml")
    convert_to_wenet_yaml(configs, wenet_train_yaml, fields_to_keep)

    wenet_model_path = os.path.join(args.output_dir, "wenet_paraformer.pt")
    convert_to_wenet_state_dict(args, wenet_model_path)

    print("Please check {} {} {} {} {} in {}".format(json_cmvn_path,
                                                     wenet_train_yaml,
                                                     wenet_model_path,
                                                     wenet_units, seg_dict,
                                                     args.output_dir))


if __name__ == "__main__":

    main()
