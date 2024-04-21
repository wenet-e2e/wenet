import argparse
import dataclasses
import os
import torch


# https://github.com/google/gemma_pytorch/blob/main/gemma/config.py#L32
@dataclasses.dataclass
class GemmaConfig:
    vocab_size: int = 256000
    # The maximum sequence length that this model might ever be used with.
    max_position_embeddings: int = 8192
    # The number of blocks in the model.
    num_hidden_layers: int = 28
    # The number of attention heads used in the attention layers of the model.
    num_attention_heads: int = 16
    # The number of key-value heads for implementing attention.
    num_key_value_heads: int = 16
    # The hidden size of the model.
    hidden_size: int = 3072
    # The dimension of the MLP representations.
    intermediate_size: int = 24576
    # The number of head dimensions.
    head_dim: int = 256
    # The epsilon used by the rms normalization layers.
    rms_norm_eps: float = 1e-6
    # The dtype of the weights.
    dtype: str = 'bfloat16'


def get_config_for_7b() -> GemmaConfig:
    return GemmaConfig()


def get_config_for_2b() -> GemmaConfig:
    return GemmaConfig(num_hidden_layers=18,
                       num_attention_heads=8,
                       num_key_value_heads=1,
                       hidden_size=2048,
                       intermediate_size=16384)


def convert_to_wenet_state_dict(w2vbert_conformer_state_dict,
                                wenet_state_dict_path, config: GemmaConfig):

    wenet_state_dict = {}

    print("==============start CKPT Conversion =========================")
    conformer_state_dict = w2vbert_conformer_state_dict
    wenet_state_dict = {}
    for name in conformer_state_dict.keys():
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
            linear_q_value = conformer_state_dict[old_name][start:offset, :]
            start = offset
            offset = offset + config.head_dim * config.num_key_value_heads
            linear_k_value = conformer_state_dict[old_name][start:offset, :]
            start = offset
            linear_v_value = conformer_state_dict[old_name][start:, :]
            wenet_state_dict[linear_q_name] = linear_q_value
            wenet_state_dict[linear_k_name] = linear_k_value
            wenet_state_dict[linear_v_name] = linear_v_value
        elif name == 'freqs_cis':
            # rope position embeding
            name = 'decoder.pos_enc.pe'
            pe = torch.view_as_real(
                conformer_state_dict[old_name].unsqueeze(0))
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

            wenet_state_dict[name] = conformer_state_dict[old_name]
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
        config = get_config_for_2b()
    else:
        config = get_config_for_7b()
    os.makedirs(args.output_dir, exist_ok=True)

    wenet_ckpt_path = os.path.join(
        args.output_dir, 'wenet_' + os.path.basename(args.gemma_ckpt))
    convert_to_wenet_state_dict(
        checkpoint["model_state_dict"],
        wenet_ckpt_path,
        config,
    )


if __name__ == '__main__':
    main()
