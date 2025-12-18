# NOTE(Mddct): This file is to convert paraformer config to wenet's train.yaml config

import argparse
import copy
import os
from typing import Dict

import torch
import yaml

from wenet.models.paraformer.convert_paraformer_to_wenet_config_and_ckpt import (
    _filter_dict_fields, convert_to_wenet_json_cmvn)
from wenet.text.sentencepiece_tokenizer import SentencepieceTokenizer


def convert_to_wenet_yaml(configs, wenet_yaml_path: str, unit_path: str,
                          tokenizer: SentencepieceTokenizer,
                          tokenizer_path) -> Dict:
    configs = copy.deepcopy(configs)
    configs['encoder'] = 'sanm_encoder_with_tp'
    configs['encoder_conf']['input_layer'] = 'paraformer_dummy'
    configs['lfr_conf'] = {'lfr_m': 7, 'lfr_n': 6}

    configs['decoder'] = None

    configs['input_dim'] = configs['lfr_conf']['lfr_m'] * 80
    # This type not use
    del configs['encoder_conf']['selfattention_layer_type'], configs[
        'encoder_conf']['pos_enc_class']
    configs['encoder_conf']['pos_enc_layer_type'] = 'abs_pos_paraformer'

    configs['ctc_conf'] = {}
    configs['ctc_conf']['ctc_blank_id'] = 0

    configs['tokenizer'] = 'sentencepiece'
    configs['tokenizer_conf'] = {}
    configs['tokenizer_conf']['model_path'] = tokenizer_path
    configs['tokenizer_conf']['special_tokens'] = {}

    with open(unit_path, 'w') as f:
        for token, i in tokenizer.symbol_table.items():
            f.write("{} {}\n".format(token, i))

    configs['tokenizer_conf']['special_tokens']['</s>'] = 2
    configs['tokenizer_conf']['special_tokens']['<s>'] = 1
    configs['tokenizer_conf']['special_tokens']['<blank>'] = 0
    configs['tokenizer_conf']['special_tokens']['<unk>'] = 0

    configs['dataset'] = 'asr_dataset'
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

    configs['grad_clip'] = 5
    configs['accum_grad'] = 1
    configs['max_epoch'] = 100
    configs['log_interval'] = 100

    configs['model_conf'] = {}
    configs['model_conf']['length_normalized_loss'] = False
    configs['model_conf']['ctc_weight'] = 1.0
    configs['model_conf']['lsm_weight'] = 0.1

    with open(wenet_yaml_path, '+w') as f:
        f.write(yaml.dump(configs))
        f.flush()
    return configs


def convert_to_wenet_state_dict(args, wenet_model_path):
    checkpoint = torch.load(args.sensevoice_model, map_location='cpu')
    torch.save(checkpoint, wenet_model_path)


def get_args():
    parser = argparse.ArgumentParser(description='load ali-sensevoice')
    parser.add_argument('--sensevoice_config',
                        default=None,
                        help='ali released SenseVoice  model\'s config')
    parser.add_argument('--sensevoice_cmvn',
                        default=None,
                        help='ali released SenseVoice model\'s cmvn')
    parser.add_argument(
        '--sensevoice_spm',
        default=None,
        help='ali released sentencepiece tokenizer\'s model path')
    parser.add_argument('--sensevoice_model',
                        default=None,
                        help='ali released sentencepiece model path')

    parser.add_argument('--output_dir',
                        default='.',
                        help="output file:\
        global_cmvn, units.txt, train.yaml, wenet_sensevoice_small.pt")
    args = parser.parse_args()
    return args


def main():

    args = get_args()
    assert os.path.exists(args.output_dir)
    with open(args.sensevoice_config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    filter_to_keep = {
        "encoder",
        "encoder_conf",
    }
    configs = _filter_dict_fields(configs, filter_to_keep)

    json_cmvn_path = os.path.join(args.output_dir, 'global_cmvn')
    convert_to_wenet_json_cmvn(args.sensevoice_cmvn, json_cmvn_path)

    wenet_units = os.path.join(args.output_dir, 'units.txt')
    tokenizer = SentencepieceTokenizer(args.sensevoice_spm)

    vocab_size = tokenizer.vocab_size()
    configs['output_dim'] = vocab_size
    configs['model'] = 'sensevoice_small'
    configs['cmvn'] = "global_cmvn"
    configs['cmvn_conf'] = {}
    configs['cmvn_conf']['is_json_cmvn'] = True
    configs['cmvn_conf']['cmvn_file'] = json_cmvn_path
    wenet_train_yaml = os.path.join(args.output_dir, "train.yaml")
    convert_to_wenet_yaml(configs, wenet_train_yaml, wenet_units, tokenizer,
                          args.sensevoice_spm)
    wenet_model_path = os.path.join(args.output_dir,
                                    "wenet_sensevoice_small.pt")
    convert_to_wenet_state_dict(args, wenet_model_path)

    print("Please check {} {} {} {}  in {}".format(json_cmvn_path,
                                                   wenet_train_yaml,
                                                   wenet_model_path,
                                                   wenet_units,
                                                   args.output_dir))


if __name__ == "__main__":

    main()
