# Copyright (c) 2025 Wenet Community. authors: Mddct(Dinghao Zhou)
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
import json
import os
import shutil

import torch
import yaml
from wenet.dataset.kaldi_io import read_mat
from wenet.text.base_tokenizer import BaseTokenizer
from wenet.text.bpe_tokenizer import BpeTokenizer


def convert_to_wenet_yaml(tokenizer: BaseTokenizer, dims, wenet_yaml_path: str,
                          symbol_table_path: str, json_cmvn_path: str,
                          bpe_model_path: str):
    configs = {}
    configs['input_dim'] = dims['idim']
    configs['output_dim'] = dims['odim']
    assert dims['odim'] == tokenizer.vocab_size(), "{} v.s. {}".format(
        dims['odim'], tokenizer.vocab_size())

    configs['encoder'] = 'firered_conformer'
    configs['encoder_conf'] = {}
    configs['encoder_conf']['gradient_checkpointing'] = True
    configs['encoder_conf']['input_layer'] = 'firered_conv2d4'
    configs['encoder_conf']['final_norm'] = False
    configs['encoder_conf']['output_size'] = dims['d_model']
    configs['encoder_conf']['attention_heads'] = dims['n_head']
    configs['encoder_conf']['linear_units'] = dims['d_inner']
    configs['encoder_conf']['num_blocks'] = dims['n_layers_enc']
    configs['encoder_conf']['dropout_rate'] = 0.1
    configs['encoder_conf']['positional_dropout_rate'] = 0.1
    configs['encoder_conf']['attention_dropout_rate'] = 0.0
    configs['encoder_conf']['normalize_before'] = True
    configs['encoder_conf']['use_dynamic_chunk'] = False
    configs['encoder_conf']['use_dynamic_left_chunk'] = False
    configs['encoder_conf']['pos_enc_layer_type'] = "rel_pos_firered"
    configs['encoder_conf']['static_chunk_size'] = -1
    configs['encoder_conf']['key_bias'] = False
    configs['encoder_conf']['value_bias'] = False
    configs['encoder_conf']['query_bias'] = False
    configs['encoder_conf']['activation_type'] = "swish"
    configs['encoder_conf']['conv_bias'] = False
    configs['encoder_conf']['conv_inner_factor'] = 4
    configs['encoder_conf']['cnn_module_kernel'] = 33
    configs['encoder_conf']['cnn_module_norm'] = 'layer_norm'
    configs['encoder_conf'][
        'selfattention_layer_type'] = 'firered_rel_selfattn'

    configs['decoder'] = 'transformer'
    configs['decoder_conf'] = {}
    configs['decoder_conf']['tie_word_embedding'] = True
    configs['decoder_conf']['gradient_checkpointing'] = True
    configs['decoder_conf']['attention_heads'] = dims['n_head']
    configs['decoder_conf']['linear_units'] = dims['d_inner']
    configs['decoder_conf']['num_blocks'] = dims['n_layers_dec']
    configs['decoder_conf']['dropout_rate'] = 0.1
    configs['decoder_conf']['positional_dropout_rate'] = 0.1
    configs['decoder_conf']['self_attention_dropout_rate'] = 0.0
    configs['decoder_conf']['src_attention_dropout_rate'] = 0.0
    configs['decoder_conf']['use_output_layer'] = True
    configs['decoder_conf']['normalize_before'] = True
    configs['decoder_conf']['src_attention'] = True
    configs['decoder_conf']['activation_type'] = "gelu"
    configs['decoder_conf']['src_key_bias'] = False
    configs['decoder_conf']['key_bias'] = False

    configs['tokenizer'] = 'bpe'
    configs['tokenizer_conf'] = {}
    configs['tokenizer_conf']['split_with_space'] = True
    configs['tokenizer_conf']['bpe_path'] = bpe_model_path
    configs['tokenizer_conf']['symbol_table_path'] = symbol_table_path
    configs['tokenizer_conf']['non_lang_syms_path'] = None
    configs['tokenizer_conf']['special_tokens'] = {}
    configs['tokenizer_conf']['special_tokens']['sos'] = 3
    configs['tokenizer_conf']['special_tokens']['eos'] = 4

    configs['ctc_conf'] = {}
    configs['ctc_conf']['ctc_blank_id'] = 0

    configs['cmvn'] = 'global_cmvn'
    configs['cmvn_conf'] = {}
    configs['cmvn_conf']['cmvn_file'] = json_cmvn_path
    configs['cmvn_conf']['is_json_cmvn'] = True

    configs['model'] = 'firered'
    configs['model_conf'] = {}
    configs['model_conf']['ctc_weight'] = 0.3
    configs['model_conf']['lsm_weight'] = 0.1
    configs['model_conf']['length_normalized_loss'] = False

    configs['dataset'] = "asr"
    configs['dataset_conf'] = {}
    configs['dataset_conf']['filter_conf'] = {}
    configs['dataset_conf']['filter_conf']['max_length'] = 409600
    configs['dataset_conf']['filter_conf']['min_length'] = 0
    configs['dataset_conf']['filter_conf']['token_max_length'] = 128
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
    configs['dataset_conf']['fbank_conf'] = {}
    configs['dataset_conf']['fbank_conf']['num_mel_bins'] = 80
    configs['dataset_conf']['fbank_conf']['frame_shift'] = 10
    configs['dataset_conf']['fbank_conf']['frame_length'] = 25
    configs['dataset_conf']['fbank_conf']['dither'] = 0.1
    configs['dataset_conf']['batch_conf'] = {}
    configs['dataset_conf']['batch_conf']['batch_type'] = 'dynamic'
    configs['dataset_conf']['batch_conf']['max_frames_in_batch'] = 12000

    configs['grad_clip'] = 1
    configs['accum_grad'] = 1
    configs['max_epoch'] = 100
    configs['log_interval'] = 100

    configs['optim'] = "adam"
    configs['optim_conf'] = {}
    configs['optim_conf']['lr'] = 0.0005
    configs['scheduler'] = "warmuplr"
    configs['scheduler_conf'] = {}
    configs['scheduler_conf']['warmup_steps'] = 12000

    with open(wenet_yaml_path, '+w') as f:
        f.write(yaml.dump(configs))
        f.flush()

    print(configs)


def convert_to_wenet_state_dict(firered_state_dict, wenet_state_dict_path):
    wenet_state_dict = {}
    unused = []
    print(
        "===================== start CKPT Conversion ========================="
    )
    for name in firered_state_dict.keys():
        if 'llm.base_model' in name:
            continue
        original_name = copy.deepcopy(name)
        if 'input_preprocessor' in original_name:
            name = name.replace("input_preprocessor", "embed")
            name = name.replace('encoder.embed.out', 'encoder.embed.out.0')

        name = name.replace("decoder.token_embedding", "decoder.embed.0")
        name = name.replace("encoder.layer_stack", "encoder.encoders")
        name = name.replace("decoder.layer_stack", "decoder.decoders")
        # decoder attn
        name = name.replace(".cross_attn.w_qs", ".src_attn.linear_q")
        name = name.replace(".cross_attn.w_ks", ".src_attn.linear_k")
        name = name.replace(".cross_attn.w_vs", ".src_attn.linear_v")
        name = name.replace(".cross_attn.fc", ".src_attn.linear_out")
        name = name.replace(".self_attn.w_qs", ".self_attn.linear_q")
        name = name.replace(".self_attn.w_ks", ".self_attn.linear_k")
        name = name.replace(".self_attn.w_vs", ".self_attn.linear_v")
        name = name.replace(".self_attn.fc", ".self_attn.linear_out")
        # encoder attn
        name = name.replace(".mhsa.w_qs", ".self_attn.linear_q")
        name = name.replace(".mhsa.w_ks", ".self_attn.linear_k")
        name = name.replace(".mhsa.w_vs", ".self_attn.linear_v")
        name = name.replace(".mhsa.fc", ".self_attn.linear_out")
        name = name.replace(".mhsa.pos_bias_u", ".self_attn.pos_bias_u")
        name = name.replace(".mhsa.pos_bias_v", ".self_attn.pos_bias_v")
        name = name.replace(".mhsa.linear_pos", ".self_attn.linear_pos")

        # decoder mlp
        name = name.replace(".mlp.", ".feed_forward.")
        # encodr mlp
        name = name.replace(".ffn1.net.1", ".feed_forward_macaron.w_1")
        name = name.replace(".ffn1.net.4", ".feed_forward_macaron.w_2")
        name = name.replace(".ffn2.net.1", ".feed_forward.w_1")
        name = name.replace(".ffn2.net.4", ".feed_forward.w_2")

        # decoder pre norm
        name = name.replace(".self_attn_norm.", ".norm1.")
        name = name.replace(".cross_attn_norm.", ".norm2.")
        name = name.replace(".mlp_norm.", ".norm3.")
        # encoder pre norm
        name = name.replace(".ffn1.net.0.", ".norm_ff_macaron.")
        name = name.replace(".mhsa.layer_norm_q.", ".self_attn.layer_norm_q.")
        name = name.replace(".mhsa.layer_norm_k.", ".self_attn.layer_norm_k.")
        name = name.replace(".mhsa.layer_norm_v.", ".self_attn.layer_norm_v.")
        name = name.replace(".conv.pre_layer_norm.", ".norm_conv.")
        name = name.replace(".ffn2.net.0", ".norm_ff")
        name = name.replace(".layer_norm.", ".norm_final.")
        name = name.replace(".layer_norm.", ".norm_final.")

        # encoder conv
        if 'embed' not in name:
            name = name.replace(".conv.", ".conv_module.")
            name = name.replace(".batch_norm.", ".norm.")

        if "decoder" in name:
            name = name.replace("cross_attn_ln", "norm2")
            name = name.replace("mlp_ln", "norm3")
        else:
            name = name.replace("mlp_ln", "norm2")

        if original_name == "decoder.tgt_word_emb.weight":
            name = "decoder.embed.0.weight"
        if original_name == "decoder.tgt_word_prj.weight":
            name = "decoder.output_layer.weight"
        if 'decoder.layer_norm_out.' in original_name:
            name = name.replace('decoder.layer_norm_out', 'decoder.after_norm')

        print("name  {} ==> {}".format(original_name, name))
        print("type  {} ==> torch.float32".format(
            firered_state_dict[original_name].dtype))
        print("shape {}\n".format(firered_state_dict[original_name].shape))
        if (original_name == name):
            unused.append(name)
        else:
            wenet_state_dict[name] = firered_state_dict[original_name].float()
    for name in unused:
        print("NOTE!!! drop {}".format(name))
    print("Saving fp32 ckpt to {}...".format(wenet_state_dict_path))
    torch.save(wenet_state_dict, wenet_state_dict_path)
    print(
        "DONE\n===================== End CKPT Conversion =========================\n"
    )


def convert_to_wenet_units(tokenizer: BaseTokenizer, units_txt_path):
    with open(units_txt_path, '+w') as f:
        for i, word in enumerate(tokenizer.symbol_table):
            f.write('{} {}\n'.format(i, word))
            f.flush()


def convert_cmvn_to_wenet_json_cmvn(firered_cmvn, units_txt_path):
    states = read_mat(firered_cmvn)
    assert states.ndim == 2
    assert states.shape[1] == 81
    frames = states[0][-1]

    states_json = {}
    states_json['mean_stat'] = states[0][:-1].tolist()
    states_json['var_stat'] = states[1][:-1].tolist()
    states_json['frame_num'] = frames

    with open(units_txt_path, 'w') as f:
        json.dump(states_json, f)


def get_args():
    parser = argparse.ArgumentParser(description='load and parse whisper')
    # yapf: disable
    parser.add_argument(
        '--firered_model_dir',
        required=True,
        help='https://huggingface.co/FireRedTeam/FireRedASR-AED-L/tree/main'
    )
    # yapf: enable
    parser.add_argument('--output_dir',
                        default='.',
                        help='output file in wenet\'s style: ' +
                        'units.txt, train.yaml, model.pt')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    checkpoint = torch.load(os.path.join(args.firered_model_dir,
                                         'model.pth.tar'),
                            map_location="cpu")

    os.makedirs(args.output_dir)

    bpe_model_path = os.path.join(args.firered_model_dir,
                                  'train_bpe1000.model')
    tokenizer = BpeTokenizer(os.path.join(args.firered_model_dir,
                                          'train_bpe1000.model'),
                             os.path.join(args.firered_model_dir, 'dict.txt'),
                             split_with_space=True)

    units_text_path = os.path.join(args.output_dir, 'units.txt')
    shutil.copy(os.path.join(args.firered_model_dir, 'dict.txt'),
                units_text_path)
    wenet_bpe_model_path = os.path.join(args.output_dir,
                                        os.path.basename(bpe_model_path))
    shutil.copy(bpe_model_path, wenet_bpe_model_path)

    firered_cmvn = os.path.join(args.firered_model_dir, 'cmvn.ark')
    wenet_json_cmvn = os.path.join(args.output_dir, 'global_cmvn')
    convert_cmvn_to_wenet_json_cmvn(firered_cmvn, wenet_json_cmvn)

    convert_to_wenet_state_dict(
        checkpoint["model_state_dict"],
        os.path.join(args.output_dir, 'wenet_firered.pt'))

    convert_to_wenet_yaml(
        tokenizer,
        vars(checkpoint["args"]),
        os.path.join(args.output_dir, 'train.yaml'),
        units_text_path,
        wenet_json_cmvn,
        wenet_bpe_model_path,
    )


if __name__ == "__main__":

    main()
