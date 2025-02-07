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
"""
Requirements:

```bash
pip install -U openai-whisper
```

Example:

```bash
# Converts the model from OpenAI to WeNet format:
python convert_whisper_to_wenet_config_and_ckpt.py \
    --whisper_ckpt large-v3.pt \
    --output_dir exp/whisper/large-v3
```
"""

import argparse
import copy
import os
import sys
import torch
import yaml

_cpath_ = sys.path[0]
sys.path.remove(_cpath_)
from whisper.tokenizer import get_tokenizer

sys.path.insert(0, _cpath_)


def convert_to_wenet_yaml(tokenizer, dims, wenet_yaml_path: str):
    configs = {}
    configs['input_dim'] = dims['n_mels']
    configs['output_dim'] = dims['n_vocab']
    assert dims['n_vocab'] == tokenizer.encoding.n_vocab, "{} v.s. {}".format(
        dims['n_vocab'], tokenizer.encoding.n_vocab)

    configs['encoder'] = 'transformer'
    configs['encoder_conf'] = {}
    configs['encoder_conf']['gradient_checkpointing'] = True
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
    configs['encoder_conf']['pos_enc_layer_type'] = "abs_pos_whisper"
    configs['encoder_conf']['static_chunk_size'] = -1
    configs['encoder_conf']['key_bias'] = False
    configs['encoder_conf']['activation_type'] = "gelu"

    configs['decoder'] = 'transformer'
    configs['decoder_conf'] = {}
    configs['decoder_conf']['tie_word_embedding'] = True
    configs['decoder_conf']['gradient_checkpointing'] = True
    configs['decoder_conf']['attention_heads'] = dims['n_text_head']
    configs['decoder_conf']['linear_units'] = dims['n_text_state'] * 4
    configs['decoder_conf']['num_blocks'] = dims['n_text_layer']
    configs['decoder_conf']['dropout_rate'] = 0.1
    configs['decoder_conf']['positional_dropout_rate'] = 0.1
    configs['decoder_conf']['self_attention_dropout_rate'] = 0.0
    configs['decoder_conf']['src_attention_dropout_rate'] = 0.0
    configs['decoder_conf']['input_layer'] = "embed_learnable_pe"
    configs['decoder_conf']['use_output_layer'] = True
    configs['decoder_conf']['normalize_before'] = True
    configs['decoder_conf']['src_attention'] = True
    configs['decoder_conf']['key_bias'] = False
    configs['decoder_conf']['src_key_bias'] = False
    configs['decoder_conf']['activation_type'] = "gelu"

    configs['tokenizer'] = 'whisper'
    configs['tokenizer_conf'] = {}
    configs['tokenizer_conf']['is_multilingual'] = dims['n_vocab'] >= 51865
    configs['tokenizer_conf']['num_languages'] = dims['n_vocab'] - 51765 - \
        int(configs['tokenizer_conf']['is_multilingual'])
    configs['tokenizer_conf']['split_with_space'] = False
    configs['tokenizer_conf']['bpe_path'] = None
    configs['tokenizer_conf']['symbol_table_path'] = None
    configs['tokenizer_conf']['non_lang_syms_path'] = None
    configs['tokenizer_conf']['special_tokens'] = {}
    configs['tokenizer_conf']['special_tokens']['sot'] = tokenizer.sot
    configs['tokenizer_conf']['special_tokens']['eot'] = tokenizer.eot
    configs['tokenizer_conf']['special_tokens'][
        'sot_prev'] = tokenizer.sot_prev
    configs['tokenizer_conf']['special_tokens'][
        'transcribe'] = tokenizer.transcribe
    configs['tokenizer_conf']['special_tokens'][
        'translate'] = tokenizer.translate
    configs['tokenizer_conf']['special_tokens'][
        'no_timestamps'] = tokenizer.no_timestamps
    configs['tokenizer_conf']['special_tokens'][
        'no_speech'] = tokenizer.no_speech
    configs['tokenizer_conf']['special_tokens']['timestamp_begin'] = \
        tokenizer.timestamp_begin

    configs['ctc_conf'] = {}
    configs['ctc_conf']['ctc_blank_id'] = tokenizer.no_speech

    configs['cmvn'] = None
    configs['cmvn_conf'] = {}
    configs['cmvn_conf']['cmvn_file'] = None
    configs['cmvn_conf']['is_json_cmvn'] = None

    configs['model'] = "whisper"
    configs['model_conf'] = {}
    configs['model_conf']['ctc_weight'] = 0.3
    configs['model_conf']['lsm_weight'] = 0.1
    configs['model_conf']['length_normalized_loss'] = False

    configs['dataset'] = "asr"
    configs['dataset_conf'] = {}
    configs['dataset_conf']['filter_conf'] = {}
    configs['dataset_conf']['filter_conf'][
        'max_length'] = dims['n_audio_ctx'] * 2  # 1/2 subsample # noqa
    configs['dataset_conf']['filter_conf']['min_length'] = 0
    configs['dataset_conf']['filter_conf']['token_max_length'] = dims[
        'n_text_ctx']
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
    configs['dataset_conf']['log_mel_spectrogram_conf']['num_mel_bins'] = dims[
        'n_mels']
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
    configs['optim_conf']['lr'] = 0.0005
    configs['scheduler'] = "warmuplr"
    configs['scheduler_conf'] = {}
    configs['scheduler_conf']['warmup_steps'] = 12000

    with open(wenet_yaml_path, '+w') as f:
        f.write(yaml.dump(configs))
        f.flush()

    print(configs)


def convert_to_wenet_state_dict(whisper_state_dict, wenet_state_dict_path):
    wenet_state_dict = {}
    unused = []
    print(
        "===================== start CKPT Conversion ========================="
    )
    for name in whisper_state_dict.keys():
        original_name = copy.deepcopy(name)
        name = name.replace("encoder.conv1", "encoder.embed.conv.0")
        name = name.replace("encoder.conv2", "encoder.embed.conv.2")
        name = name.replace("decoder.token_embedding", "decoder.embed.0")
        name = name.replace("encoder.blocks", "encoder.encoders")
        name = name.replace("decoder.blocks", "decoder.decoders")
        name = name.replace(".cross_attn.query", ".src_attn.linear_q")
        name = name.replace(".cross_attn.key", ".src_attn.linear_k")
        name = name.replace(".cross_attn.value", ".src_attn.linear_v")
        name = name.replace(".cross_attn.out", ".src_attn.linear_out")
        name = name.replace(".attn.query", ".self_attn.linear_q")
        name = name.replace(".attn.key", ".self_attn.linear_k")
        name = name.replace(".attn.value", ".self_attn.linear_v")
        name = name.replace(".attn.out", ".self_attn.linear_out")
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
        if original_name == "decoder.positional_embedding":
            whisper_state_dict[name] = whisper_state_dict[name].unsqueeze(0)
            name = "decoder.embed.1.pe"
        elif original_name == "encoder.positional_embedding":
            whisper_state_dict[name] = whisper_state_dict[name].unsqueeze(0)
            name = "encoder.embed.pos_enc.pe"
        print("name  {} ==> {}".format(original_name, name))
        print("type  {} ==> torch.float32".format(
            whisper_state_dict[original_name].dtype))
        print("shape {}\n".format(whisper_state_dict[original_name].shape))
        if (original_name == name):
            unused.append(name)
        else:
            wenet_state_dict[name] = whisper_state_dict[original_name].float()
    for name in unused:
        print("NOTE!!! drop {}".format(name))
    print("Saving fp32 ckpt to {}...".format(wenet_state_dict_path))
    torch.save(wenet_state_dict, wenet_state_dict_path)
    print(
        "DONE\n===================== End CKPT Conversion =========================\n"
    )


def convert_to_wenet_units(tokenizer, units_txt_path):
    """ NOTE(xcsong):
        The "units.txt" file is solely for adapting to the training API of Wenet
        and for quickly checking the corresponding text of an ID when necessary.
        It does not play any role in the tokenization process,
        which is carried out by the tokenizer of openai-whisper.
    """
    n_vocab = tokenizer.encoding.n_vocab
    with open(units_txt_path, "+w") as f:
        for i in range(n_vocab):
            unit = str(tokenizer.encoding.decode_single_token_bytes(i))
            if len(unit) == 0:
                unit = str(i)
                print("can not decode id {}, convert to str({})".format(i, i))
            unit = unit.replace(" ", "<space>")
            f.write("{} {}\n".format(unit, i))
            f.flush()


def get_args():
    parser = argparse.ArgumentParser(description='load and parse whisper')
    # yapf: disable
    parser.add_argument(
        '--whisper_ckpt',
        required=True,
        help='https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt'  # noqa
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
    checkpoint = torch.load(args.whisper_ckpt, map_location="cpu")
    multilingual = checkpoint["dims"]['n_vocab'] >= 51865
    num_languages = checkpoint["dims"]['n_vocab'] - 51765 - int(multilingual)
    tokenizer = get_tokenizer(multilingual=multilingual,
                              num_languages=num_languages)

    convert_to_wenet_state_dict(
        checkpoint["model_state_dict"],
        os.path.join(args.output_dir, 'wenet_whisper.pt'))
    convert_to_wenet_units(tokenizer, os.path.join(args.output_dir,
                                                   'units.txt'))
    convert_to_wenet_yaml(tokenizer, checkpoint["dims"],
                          os.path.join(args.output_dir, 'train.yaml'))


if __name__ == "__main__":

    main()
