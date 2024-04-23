import argparse
import os
from typing import Dict
import torch
import yaml

from wenet.LLM.script.config import (Config, gemma_config_for_2b,
                                     gemma_config_for_7b)
from wenet.text.hugging_face_tokenizer import HuggingFaceTokenizer


def wenet_llm_tokenizer_conf(config: Config, tokenizer_path: str) -> Dict:
    tokenizer = HuggingFaceTokenizer(tokenizer_path)
    assert config.vocab_size == tokenizer.vocab_size()
    bos = tokenizer.tokens2ids(["<bos>"])[0]
    eos = tokenizer.tokens2ids(["<eos>"])[0]
    unk = tokenizer.tokens2ids(["<pad>"])[0]
    configs = {}
    configs['tokenizer'] = 'huggingface'
    configs['tokenizer_conf'] = {}
    configs['tokenizer_conf']['model_path'] = tokenizer_path
    configs['tokenizer_conf']['special_tokens'] = {}
    configs['tokenizer_conf']['special_tokens']['<bos>'] = bos
    configs['tokenizer_conf']['special_tokens']['<eos>'] = eos
    configs['tokenizer_conf']['special_tokens']['<pad>'] = unk
    return configs


def wenet_llm_dataset_and_train_conf(config: Config) -> Dict:
    configs = {}
    configs['dataset'] = "llm"
    configs['dataset_conf'] = {}
    configs['dataset_conf']['filter_conf'] = {}
    configs['dataset_conf']['filter_conf'][
        'token_max_length'] = config.max_position_embeddings
    configs['dataset_conf']['filter_conf']['token_min_length'] = 1
    configs['dataset_conf']['shuffle'] = True
    configs['dataset_conf']['shuffle_conf'] = {}
    configs['dataset_conf']['shuffle_conf']['shuffle_size'] = 1500
    configs['dataset_conf']['sort'] = True
    configs['dataset_conf']['sort_conf'] = {}
    configs['dataset_conf']['sort_conf']['sort_size'] = 500
    configs['dataset_conf']['batch_conf'] = {}
    configs['dataset_conf']['batch_conf']['batch_type'] = 'dynamic'
    configs['dataset_conf']['batch_conf']['max_frames_in_batch'] = 12000

    configs['dataset_conf']['data_style'] = 'sft'
    configs['dataset_conf']['data_style_conf'] = {}
    configs['dataset_conf']['data_style_conf']['add_bos'] = True
    configs['dataset_conf']['data_style_conf']['add_eos'] = True
    configs['dataset_conf']['data_style_conf']['template'] = 'gemma'

    configs['grad_clip'] = 5
    configs['accum_grad'] = 4
    configs['max_epoch'] = 100
    configs['log_interval'] = 100
    configs['save_interval'] = 3000

    configs['optim'] = "adam"
    configs['optim_conf'] = {}
    configs['optim_conf']['lr'] = 0.0005
    configs['scheduler'] = "warmuplr"
    configs['scheduler_conf'] = {}
    configs['scheduler_conf']['warmup_steps'] = 12000
    return configs


def convert_to_wenet_yaml(config: Config, wenet_yaml_path: str,
                          tokenizer_path):
    configs = {}
    configs.update(wenet_llm_tokenizer_conf(config, tokenizer_path))
    configs['output_dim'] = config.vocab_size
    configs['decoder'] = 'decoder_only'
    configs['decoder_conf'] = config.to_wenet_config()
    configs['decoder_conf']['dropout_rate'] = 0.0
    configs['decoder_conf']['attention_dropout_rate'] = 0.0
    configs['decoder_conf']['positional_dropout_rate'] = 0.0
    configs['decoder_conf']['gradient_checkpointing'] = True
    configs['decoder_conf']['normalize_before'] = True

    configs['model'] = "causal_lm"
    configs['model_conf'] = {}
    configs['model_conf']['linear_bias'] = False
    configs['model_conf']['tie_word_embedding'] = True
    configs['model_conf']['lsm_weight'] = 0.1
    configs['model_conf']['length_normalized_loss'] = False

    configs.update(wenet_llm_dataset_and_train_conf(config))

    with open(wenet_yaml_path, '+w') as f:
        f.write(yaml.dump(configs))
        f.flush()

    print(configs)


def convert_to_wenet_state_dict(gemma_state_dict, wenet_state_dict_path,
                                config: Config):

    print("==============start CKPT Conversion =========================")
    wenet_state_dict = {}
    for name in gemma_state_dict.keys():
        old_name = name
        # embed
        name = name.replace('embedder.weight', 'embed.weight')

        # layers to decoders
        name = name.replace('model.layers', 'decoder.decoders')

        if 'self_attn.qkv_proj' in name:
            # att weight
            i_layer = name.split('.')[2]
            layer_prefix = 'decoder.decoders.' + i_layer
            linear_q_name = layer_prefix + '.self_attn.linear_q.weight'
            linear_k_name = layer_prefix + '.self_attn.linear_k.weight'
            linear_v_name = layer_prefix + '.self_attn.linear_v.weight'

            start = 0
            offset = config.num_attention_heads * config.head_dim
            linear_q_value = gemma_state_dict[old_name][start:offset, :]
            start = offset
            offset = offset + config.head_dim * config.num_key_value_heads
            linear_k_value = gemma_state_dict[old_name][start:offset, :]
            start = offset
            linear_v_value = gemma_state_dict[old_name][start:, :]
            wenet_state_dict[linear_q_name] = linear_q_value
            wenet_state_dict[linear_k_name] = linear_k_value
            wenet_state_dict[linear_v_name] = linear_v_value
        elif name == 'freqs_cis':
            # rope position embeding
            name = 'decoder.pos_enc.pe'
            pe = torch.view_as_real(gemma_state_dict[old_name].unsqueeze(0))
            wenet_state_dict[name] = pe
        else:
            # att out dim
            name = name.replace('self_attn.o_proj', 'self_attn.linear_out')

            # mlp
            name = name.replace('mlp.gate_proj', 'feed_forward.gate')
            name = name.replace('mlp.up_proj', 'feed_forward.w_1')
            name = name.replace('mlp.down_proj', 'feed_forward.w_2')

            # pre ln (rms norm)
            name = name.replace('input_layernorm', 'norm1')
            # before mlp ln: (rms norm)
            name = name.replace('post_attention_layernorm', 'norm2')
            # final norm
            name = name.replace('model.norm.weight',
                                'decoder.final_norm.weight')

            wenet_state_dict[name] = gemma_state_dict[old_name]
    # NOTE(Mddct): tie weight
    wenet_state_dict['out.weight'] = wenet_state_dict['embed.weight']
    print("Saving {} ckpt to {}...".format(config.dtype,
                                           wenet_state_dict_path))
    torch.save(wenet_state_dict, wenet_state_dict_path)
    print(
        "DONE\n===================- End CKPT Conversion ====================\n"
    )


def get_args():
    parser = argparse.ArgumentParser(
        description='load and convert google gemma ckpt')
    parser.add_argument(
        '--gemma_ckpt',
        required=True,
        help='https://www.kaggle.com/models/google/gemma/frameworks/pyTorch')
    parser.add_argument('--model_size', type=str, required=True)
    parser.add_argument('--output_dir',
                        default='.',
                        help='output file in wenet\'s style')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    args.jit = True
    checkpoint = torch.load(args.gemma_ckpt, map_location="cpu")
    model_size = args.model_size
    assert model_size in ["2b", "7b"]

    if model_size == '2b':
        config = gemma_config_for_2b()
        args.gemma_tokenizer = 'google/gemma-2b'
    else:
        config = gemma_config_for_7b()
        args.gemma_tokenizer = 'google/gemma-7b'
    os.makedirs(args.output_dir, exist_ok=True)

    wenet_ckpt_path = os.path.join(
        args.output_dir, 'wenet_' + os.path.basename(args.gemma_ckpt))
    convert_to_wenet_state_dict(
        checkpoint["model_state_dict"],
        wenet_ckpt_path,
        config,
    )
    wenet_yaml_path = os.path.join(args.output_dir, 'train.yaml')
    convert_to_wenet_yaml(
        config,
        wenet_yaml_path,
        args.gemma_tokenizer,
    )


if __name__ == '__main__':
    main()
