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
    configs['input_dim'] = dims['n_mels']
    configs['output_dim'] = dims['n_vocab']

    configs['encoder'] = 'transformer'
    configs['encoder_conf'] = {}
    configs['encoder_conf']['input_layer'] = 'conv1d2'
    configs['encoder_conf']['output_size'] = dims['n_audio_state']
    configs['encoder_conf']['attention_heads'] = dims['n_audio_head']
    configs['encoder_conf']['linear_units'] = dims['n_audio_state'] * 4
    configs['encoder_conf']['num_blocks'] = dims['n_audio_layer']
    configs['encoder_conf']['dropout_rate'] = 0.1
    configs['encoder_conf']['positional_dropout_rate'] = 0.1
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
    configs['decoder_conf']['dropout_rate'] = 0.1
    configs['decoder_conf']['positional_dropout_rate'] = 0.1
    configs['decoder_conf']['self_attention_dropout_rate'] = 0.0
    configs['decoder_conf']['src_attention_dropout_rate'] = 0.0
    configs['decoder_conf']['input_layer'] = "embed_nope"
    configs['decoder_conf']['use_output_layer'] = False
    configs['decoder_conf']['normalize_before'] = True
    configs['decoder_conf']['src_attention'] = True
    configs['decoder_conf']['key_bias'] = False

    configs['model_conf'] = {}
    configs['model_conf']['ctc_weight'] = 0.3
    configs['model_conf']['lsm_weight'] = 0.1
    configs['model_conf']['length_normalized_loss'] = False

    configs['dataset_conf'] = {}
    configs['dataset_conf']['filte_conf'] = {}
    configs['dataset_conf']['filte_conf']['max_length'] = dims['n_audio_ctx'] * 2  # 1/2 subsample, noqa
    configs['dataset_conf']['filte_conf']['min_length'] = 0
    configs['dataset_conf']['filte_conf']['token_max_length'] = dims['n_text_ctx']
    configs['dataset_conf']['filte_conf']['token_min_length'] = 1
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

    configs['optim'] = "adam"
    configs['optim_conf'] = {}
    configs['optim_conf']['lr'] = 0.002
    configs['scheduler'] = "warmuplr"
    configs['scheduler_conf'] = {}
    configs['scheduler_conf']['warmup_steps'] = 25000

    with open(wenet_yaml_path, '+w') as f:
        f.write(yaml.dump(configs))
        f.flush()


def convert_to_wenet_state_dict(whisper_state_dict):
    wenet_state_dict = {}
    unused = []
    print("========================== start CKPT Conversion ==============================")
    for name in whisper_state_dict.keys():
        original_name = copy.deepcopy(name)
        name = name.replace("encoder.conv1", "encoder.embed.conv.0")
        name = name.replace("encoder.conv2", "encoder.embed.conv.2")
        name = name.replace("decoder.token_embedding", "decoder.embed.0")
        name = name.replace("encoder.blocks", "encoder.encoders")
        name = name.replace("decoder.blocks", "decoder.decoders")
        name = name.replace("cross_attn.query", "src_attn.linear_q")
        name = name.replace("cross_attn.key", "src_attn.linear_k")
        name = name.replace("cross_attn.value", "src_attn.linear_v")
        name = name.replace("cross_attn.out", "src_attn.linear_out")
        name = name.replace("attn.query", "self_attn.linear_q")
        name = name.replace("attn.key", "self_attn.linear_k")
        name = name.replace("attn.value", "self_attn.linear_v")
        name = name.replace("attn.out", "self_attn.linear_out")
        name = name.replace("mlp.0", "feed_forward.w_1")
        name = name.replace("mlp.2", "feed_forward.w_2")
        if "decoder" in name:
            name = name.replace("cross_attn_ln", "norm2")
            name = name.replace("mlp_ln", "norm3")
        else:
            name = name.replace("mlp_ln", "norm2")
        name = name.replace("attn_ln", "norm1")
        name = name.replace("encoder.ln_post", "encoder.after_norm")
        name = name.replace("decoder.ln", "decoder.after_norm")
        print("name  {} ==> {}".format(original_name, name))
        print("type  {} ==> torch.float32".format(
            whisper_state_dict[original_name].dtype))
        print("shape {}\n".format(whisper_state_dict[original_name].shape))
        if (original_name == name):
            unused.append(name)
        else:
            wenet_state_dict[name] = whisper_state_dict[original_name].float()
    print("========================== End CKPT Conversion ==============================\n")
    for name in unused:
        print("NOTE!!! drop {}".format(name))
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
