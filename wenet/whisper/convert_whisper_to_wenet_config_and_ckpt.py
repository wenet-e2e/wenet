# Copyright (c) 2023 Wenet Community. (authors: Xingchen Song)
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

import argparse
import copy
import os
import torch
import yaml


def convert_to_wenet_yaml(dims, wenet_yaml_path: str):
    configs = {}
    configs['whisper'] = True
    configs['input_dim'] = 128
    configs['output_dim'] = dims['n_vocab']

    configs['encoder'] = 'transformer'
    configs['encoder_conf'] = {}
    configs['encoder_conf']['input_layer'] = 'conv1d2'
    configs['encoder_conf']['output_size'] = dims['n_audio_state']
    configs['encoder_conf']['attention_heads'] = dims['n_audio_head']
    configs['encoder_conf']['linear_units'] = dims['n_audio_state'] * 4
    configs['encoder_conf']['num_blocks'] = dims['n_audio_layer']
    configs['encoder_conf']['dropout_rate'] = 0.0
    configs['encoder_conf']['positional_dropout_rate'] = 0.0
    configs['encoder_conf']['attention_dropout_rate'] = 0.0
    configs['encoder_conf']['normalize_before'] = True
    configs['encoder_conf']['use_dynamic_chunk'] = False
    configs['encoder_conf']['use_dynamic_left_chunk'] = False
    configs['encoder_conf']['pos_enc_layer_type'] = "abs_pos"
    configs['encoder_conf']['static_chunk_size'] = -1
    configs['encoder_conf']['key_bias'] = False

    configs['decoder'] = 'transformer'
    configs['decoder_conf'] = {}
    configs['decoder_conf']['attention_heads'] = dims['n_text_head']
    configs['decoder_conf']['linear_units'] = dims['n_text_state'] * 4
    configs['decoder_conf']['num_blocks'] = dims['n_text_layer']
    configs['decoder_conf']['dropout_rate'] = 0.0
    configs['decoder_conf']['positional_dropout_rate'] = 0.0
    configs['decoder_conf']['self_attention_dropout_rate'] = 0.0
    configs['decoder_conf']['src_attention_dropout_rate'] = 0.0
    configs['decoder_conf']['key_bias'] = False

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


def convert_to_wenet_state_dict(whisper_state_dict):
    wenet_state_dict = {}
    for name in whisper_state_dict.keys():
        original_name = copy.deepcopy(name)
        name.replace("encoder.blocks", "encoder.encoders")
        name.replace("decoder.blocks", "decoder.decoders")
        name.replace("attn.query", "self_attn.linear_q")
        name.replace("attn.key", "self_attn.linear_k")
        name.replace("attn.value", "self_attn.linear_v")
        name.replace("attn.out", "self_attn.linear_out")
        name.replace("mlp.0", "feed_forward.w_1")
        name.replace("mlp.2", "feed_forward.w_2")
        name.replace("cross_attn.query", "src_attn.linear_q")
        name.replace("cross_attn.key", "src_attn.linear_k")
        name.replace("cross_attn.value", "src_attn.linear_v")
        name.replace("cross_attn.out", "src_attn.linear_out")
        name.replace("attn_ln", "norm1")
        if "decoder" in name:
            name.replace("cross_attn_ln", "norm2")
            name.replace("mlp_ln", "norm3")
        else:
            name.replace("mlp_ln", "norm2")
        print("name {} ==> {}".format(original_name, name))
        print("type {} ==> torch.float32\n".format(
            whisper_state_dict[original_name].dtype))
        wenet_state_dict[name] = whisper_state_dict[original_name].float()
    return wenet_state_dict


def extract_dict(whisper_units, units_txt_path):
    pass


def get_args():
    parser = argparse.ArgumentParser(description='load and parse whisper')
    parser.add_argument('--whisper_ckpt', required=True,
                        help='https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt')
    parser.add_argument('--whisper_units', required=True,
                        help='https://github.com/openai/whisper/blob/main/whisper/assets/multilingual.tiktoken')
    parser.add_argument('--output_dir', default='.',
                        help='output file in wenet\'s style: ' +
                             'units.txt, train.yaml, model.pt')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    checkpoint = torch.load(args.whisper_ckpt, map_location="cpu")
    dims = checkpoint["dims"]
    whisper_state_dict = checkpoint["model_state_dict"]
    wenet_state_dict = convert_to_wenet_state_dict(whisper_state_dict)

    vocab_size = extract_dict(args.whisper_units,
                              os.path.join(args.output_dir, 'units.txt'))
    convert_to_wenet_yaml(dims, os.path.join(args.output_dir, 'train.yaml'))


if __name__ == "__main__":

    main()
