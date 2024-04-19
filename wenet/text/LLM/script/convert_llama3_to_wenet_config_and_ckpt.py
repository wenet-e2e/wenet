import argparse
import dataclasses
import os
import torch


# https://github.com/google/gemma_pytorch/blob/main/gemma/config.py#L32
@dataclasses.dataclass
class LlamaConfig:
    vocab_size: int = 256000
    # The maximum sequence length that this model might ever be used with.
    max_position_embeddings: int = 2048
    # The number of blocks in the model.
    num_hidden_layers: int = 32
    # The number of attention heads used in the attention layers of the model.
    num_attention_heads: int = 32
    # The number of key-value heads for implementing attention.
    num_key_value_heads: int = 8
    # The hidden size of the model.
    hidden_size: int = 4096
    # The dimension of the MLP representations.
    intermediate_size: int = 14336
    # The number of head dimensions.
    head_dim: int = 128
    # The epsilon used by the rms normalization layers.
    rms_norm_eps: float = 1e-5
    # tope theta
    rope_theta: float = 500000.0
    # The dtype of the weights.
    dtype: str = 'bfloat16'


def get_config_for_8b() -> LlamaConfig:
    return LlamaConfig()


def get_config_for_70b() -> LlamaConfig:
    raise NotImplementedError('not implement')


def convert_to_wenet_state_dict(Llama3_state_dict, wenet_state_dict_path,
                                config: LlamaConfig):

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
        config = get_config_for_8b()
    else:
        config = get_config_for_70b()
    os.makedirs(args.output_dir, exist_ok=True)

    wenet_ckpt_path = os.path.join(
        args.output_dir, 'wenet_' + os.path.basename(args.llama_ckpt))
    wenet_ckpt_path = os.path.splitext(wenet_ckpt_path)[0] + ".pt"
    print(wenet_ckpt_path)
    convert_to_wenet_state_dict(
        checkpoint,
        wenet_ckpt_path,
        config,
    )


if __name__ == '__main__':
    main()
