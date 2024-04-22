import argparse
import os
import torch
import yaml

from wenet.LLM.script.config import (Config, llama3_config_for_70b,
                                     llama3_config_for_8b)
from wenet.LLM.script.convert_gemma_to_wenet_config_and_ckpt import (
    wenet_llm_dataset_and_train_conf)


def convert_to_wenet_yaml(config: Config, wenet_yaml_path: str):
    configs = {}
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
    configs['model_conf']['tie_word_embedding'] = False
    configs['model_conf']['lsm_weight'] = 0.1
    configs['model_conf']['length_normalized_loss'] = False

    configs.update(wenet_llm_dataset_and_train_conf(config))
    with open(wenet_yaml_path, '+w') as f:
        f.write(yaml.dump(configs))
        f.flush()

    print(configs)


def convert_to_wenet_state_dict(Llama3_state_dict, wenet_state_dict_path,
                                config: Config):

    wenet_state_dict = {}

    print("==============start CKPT Conversion =========================")
    conformer_state_dict = Llama3_state_dict
    wenet_state_dict = {}
    for name in conformer_state_dict.keys():
        old_name = name
        # embed
        name = name.replace('tok_embeddings.weight', 'embed.weight')

        # output
        name = name.replace('output.weight', 'out.weight')
        # layers to decoders
        name = name.replace('layers', 'decoder.decoders')

        if 'attention' in name:
            # pre ln (rms norm)
            name = name.replace('attention_norm', 'norm1')
            # att weight
            name = name.replace('.attention.wq.weight',
                                '.self_attn.linear_q.weight')
            name = name.replace('.attention.wk.weight',
                                '.self_attn.linear_k.weight')
            name = name.replace('.attention.wv.weight',
                                '.self_attn.linear_v.weight')
            # att out dim
            name = name.replace('attention.wo', 'self_attn.linear_out')
        elif name == 'norm_weight':
            name = name.replace('norm_weight', 'decoder.final_norm.weight')
        else:

            # mlp
            name = name.replace('feed_forward.w1', 'feed_forward.gate')
            name = name.replace('feed_forward.w3', 'feed_forward.w_1')
            name = name.replace('feed_forward.w2', 'feed_forward.w_2')

            # before mlp ln: (rms norm)
            name = name.replace('ffn_norm', 'norm2')
            # final norm
            name = name.replace('model.norm.weight',
                                'decoder.final_norm.weight')

        wenet_state_dict[name] = conformer_state_dict[old_name]
    print("Saving {} ckpt to {}...".format(config.dtype,
                                           wenet_state_dict_path))
    torch.save(wenet_state_dict, wenet_state_dict_path)
    print(
        "DONE\n===================- End CKPT Conversion ====================\n"
    )


def get_args():
    parser = argparse.ArgumentParser(
        description='load and convert google gemma ckpt')
    parser.add_argument('--llama_ckpt',
                        required=True,
                        help='https://llama.meta.com/llama-downloads/')
    parser.add_argument('--model_size', type=str, required=True)
    parser.add_argument('--output_dir',
                        default='.',
                        help='output file in wenet\'s style')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    args.jit = True
    checkpoint = torch.load(args.llama_ckpt, map_location="cpu")
    model_size = args.model_size
    assert model_size in ["8b", "70b"]

    if model_size == '8b':
        config = llama3_config_for_8b()
    else:
        config = llama3_config_for_70b()
    os.makedirs(args.output_dir, exist_ok=True)

    wenet_ckpt_path = os.path.join(
        args.output_dir, 'wenet_' + os.path.basename(args.llama_ckpt))
    wenet_ckpt_path = os.path.splitext(wenet_ckpt_path)[0] + ".pt"
    convert_to_wenet_state_dict(
        checkpoint,
        wenet_ckpt_path,
        config,
    )
    wenet_yaml_path = os.path.join(args.output_dir, 'train.yaml')
    convert_to_wenet_yaml(
        config,
        wenet_yaml_path,
    )


if __name__ == '__main__':
    main()
