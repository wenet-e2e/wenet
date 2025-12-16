# Copyright (c) 2025 Chengdong Liang(liangchengdongd@qq.com)
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
from transformers import AutoTokenizer, Qwen3OmniMoeForConditionalGeneration


def convert_to_wenet_yaml(tokenizer, wenet_yaml_path: str):
    configs = {}
    configs['input_dim'] = 128
    configs['output_dim'] = 151676
    configs['dtype'] = 'bf16'

    configs['encoder'] = 'transformer'
    configs['encoder_conf'] = {}
    configs['encoder_conf']['gradient_checkpointing'] = True
    configs['encoder_conf']['input_layer'] = 'qwen3omni_aut_conv2d8'
    configs['encoder_conf']['input_layer_hidden_size'] = 480
    configs['encoder_conf']['output_size'] = 1280
    configs['encoder_conf']['attention_heads'] = 20
    configs['encoder_conf']['linear_units'] = 5120
    configs['encoder_conf']['num_blocks'] = 32
    configs['encoder_conf']['dropout_rate'] = 0.1
    configs['encoder_conf']['positional_dropout_rate'] = 0.1
    configs['encoder_conf']['attention_dropout_rate'] = 0.0
    configs['encoder_conf']['normalize_before'] = True
    configs['encoder_conf']['use_dynamic_chunk'] = False
    configs['encoder_conf']['use_dynamic_left_chunk'] = False
    configs['encoder_conf']['pos_enc_layer_type'] = "abs_pos_whisper"
    configs['encoder_conf']['static_chunk_size'] = -1
    configs['encoder_conf']['activation_type'] = "gelu"

    configs['tokenizer'] = 'huggingface'
    configs['tokenizer_conf'] = {}
    configs['tokenizer_conf']['model'] = 'Qwen/Qwen3-Omni-30B-A3B-Instruct'

    configs['ctc_conf'] = {}
    configs['ctc_conf']['ctc_blank_id'] = tokenizer.pad_token_id

    configs['cmvn'] = None
    configs['cmvn_conf'] = {}
    configs['cmvn_conf']['cmvn_file'] = None
    configs['cmvn_conf']['is_json_cmvn'] = None

    configs['model'] = "qwen3omni_aut"
    configs['model_conf'] = {}
    configs['model_conf']['ctc_weight'] = 1.0
    configs['model_conf']['lsm_weight'] = 0.1
    configs['model_conf']['length_normalized_loss'] = False

    configs['dataset'] = "asr"
    configs['dataset_conf'] = {}
    configs['dataset_conf']['filter_conf'] = {}
    configs['dataset_conf']['filter_conf'][
        'max_length'] = 3000
    configs['dataset_conf']['filter_conf']['min_length'] = 0
    configs['dataset_conf']['filter_conf']['token_max_length'] = 300
    configs['dataset_conf']['filter_conf']['token_min_length'] = 1
    configs['dataset_conf']['resample_conf'] = {}
    configs['dataset_conf']['resample_conf']['resample_rate'] = 16000
    # NOTE: Disable speed_perturb, https://github.com/wenet-e2e/wenet/issues/2171
    configs['dataset_conf']['speed_perturb'] = False
    configs['dataset_conf']['spec_aug'] = True
    configs['dataset_conf']['spec_aug_conf'] = {}
    configs['dataset_conf']['spec_aug_conf']['num_t_mask'] = 2
    configs['dataset_conf']['spec_aug_conf']['num_f_mask'] = 2
    configs['dataset_conf']['spec_aug_conf']['max_t'] = 50
    configs['dataset_conf']['spec_aug_conf']['max_f'] = 10
    configs['dataset_conf']['spec_sub'] = True
    configs['dataset_conf']['spec_sub_conf'] = {}
    configs['dataset_conf']['spec_sub_conf']['num_t_sub'] = 3
    configs['dataset_conf']['spec_sub_conf']['max_t'] = 30
    configs['dataset_conf']['spec_trim'] = False
    configs['dataset_conf']['shuffle'] = True
    configs['dataset_conf']['shuffle_conf'] = {}
    configs['dataset_conf']['shuffle_conf']['shuffle_size'] = 1500
    configs['dataset_conf']['sort'] = True
    configs['dataset_conf']['sort_conf'] = {}
    configs['dataset_conf']['sort_conf']['sort_size'] = 500
    configs['dataset_conf']['feats_type'] = "log_mel_spectrogram"
    configs['dataset_conf']['log_mel_spectrogram_conf'] = {}
    configs['dataset_conf']['log_mel_spectrogram_conf']['n_fft'] = 400
    configs['dataset_conf']['log_mel_spectrogram_conf']['hop_length'] = 160
    configs['dataset_conf']['log_mel_spectrogram_conf']['num_mel_bins'] = 128
    configs['dataset_conf']['log_mel_spectrogram_conf']['padding'] = 0
    configs['dataset_conf']['batch_conf'] = {}
    configs['dataset_conf']['batch_conf']['batch_type'] = 'dynamic'
    configs['dataset_conf']['batch_conf']['batch_size'] = 26
    configs['dataset_conf']['batch_conf']['max_frames_in_batch'] = 12000
    configs['dataset_conf']['language_conf'] = {}
    configs['dataset_conf']['language_conf']['limited_langs'] = ['zh']

    configs['grad_clip'] = 5
    configs['accum_grad'] = 4
    configs['max_epoch'] = 100
    configs['log_interval'] = 100

    configs['optim'] = "adam"
    configs['optim_conf'] = {}
    configs['optim_conf']['lr'] = 0.001
    configs['scheduler'] = "warmuplr"
    configs['scheduler_conf'] = {}
    configs['scheduler_conf']['warmup_steps'] = 12000

    with open(wenet_yaml_path, '+w') as f:
        f.write(yaml.dump(configs))
        f.flush()

    print(configs)


def convert_to_wenet_state_dict(ori_state_dict,
                                wenet_state_dict_path,
                                bf16=False):
    wenet_state_dict = {}
    unused = []
    print(
        "===================== start CKPT Conversion ========================="
    )
    for name in ori_state_dict.keys():
        original_name = copy.deepcopy(name)
        name = name.replace("conv2d1.", "encoder.embed.conv.0.")
        name = name.replace("conv2d2.", "encoder.embed.conv.2.")
        name = name.replace("conv2d3.", "encoder.embed.conv.4.")
        name = name.replace("conv_out.", "encoder.embed.linear.")
        name = name.replace("layers.", "encoder.encoders.")
        name = name.replace("k_proj.", "linear_k.")
        name = name.replace("q_proj.", "linear_q.")
        name = name.replace("v_proj.", "linear_v.")
        name = name.replace("out_proj.", "linear_out.")
        name = name.replace("self_attn_layer_norm.", "norm1.")
        name = name.replace("fc1.", "feed_forward.w_1.")
        name = name.replace("fc2.", "feed_forward.w_2.")
        name = name.replace("final_layer_norm.", "norm2.")
        name = name.replace("ln_post.", "encoder.after_norm.")
        print("name  {} ==> {}".format(original_name, name))
        print("type  {} ==> torch.float32".format(
            ori_state_dict[original_name].dtype))
        print("shape {}\n".format(ori_state_dict[original_name].shape))
        if (original_name == name):
            unused.append(name)
        else:
            wenet_state_dict[name] = ori_state_dict[original_name].float()
    for name in unused:
        print("NOTE!!! drop {}".format(name))
    if bf16:
        for k, v in wenet_state_dict.items():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                wenet_state_dict[k] = v.to(torch.bfloat16)
    print("Saving ckpt to {}...".format(wenet_state_dict_path))
    torch.save(wenet_state_dict, wenet_state_dict_path)
    print(
        "===================== End CKPT Conversion =========================\n"
    )



def get_args():
    parser = argparse.ArgumentParser(description='load and parse qwen3omni_aut')
    parser.add_argument('--qwen3omni_model_dir', required=True,)
    parser.add_argument('--bf16', action='store_true', help='save bf16 model')
    parser.add_argument('--output_dir',
                        default='.',
                        help='output file in wenet\'s style: ' +
                        'units.txt, train.yaml, model.pt')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    aut_model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        args.qwen3omni_model_dir,
        dtype="auto",
        device_map="auto",
    ).thinker.audio_tower
    tokenizer = AutoTokenizer.from_pretrained(
        args.qwen3omni_model_dir,
        use_fast=False,
    )
    convert_to_wenet_state_dict(aut_model.state_dict(),
                                os.path.join(args.output_dir, 'final.pt'),
                                args.bf16)
    convert_to_wenet_yaml(tokenizer, wenet_yaml_path=os.path.join(
        args.output_dir, 'train.yaml'))
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
