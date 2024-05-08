import torch

from wenet.LLM.script.config import Config
from wenet.text.hugging_face_tokenizer import HuggingFaceTokenizer


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


def gemma_special_tokens(tokenizer_path, config: Config):
    tokenizer = HuggingFaceTokenizer(tokenizer_path)
    assert config.vocab_size == tokenizer.vocab_size()
    special_tokens = {}
    bos = tokenizer.tokens2ids(["<bos>"])[0]
    eos = tokenizer.tokens2ids(["<eos>"])[0]
    unk = tokenizer.tokens2ids(["<pad>"])[0]
    special_tokens = {}
    special_tokens['<bos>'] = bos
    special_tokens['<eos>'] = eos
    special_tokens['<pad>'] = unk
    return special_tokens
