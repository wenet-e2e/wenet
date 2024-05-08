from typing import Dict
import torch
from wenet.LLM.script.config import Config

from wenet.text.hugging_face_tokenizer import HuggingFaceTokenizer


def llama3_special_tokens(tokenizer_path, config: Config) -> Dict:
    tokenizer = HuggingFaceTokenizer(tokenizer_path)
    assert config.vocab_size == tokenizer.vocab_size()
    # "<|reserved_special_token_0|>",
    # "<|reserved_special_token_1|>",
    # "<|reserved_special_token_2|>",
    # "<|reserved_special_token_3|>",
    shi = tokenizer.tokens2ids(["<|start_header_id|>"])[0]
    ehi = tokenizer.tokens2ids(["<|end_header_id|>"])[0]
    bos = tokenizer.tokens2ids(["<|begin_of_text|>"])[0]
    eos = tokenizer.tokens2ids(["<|end_of_text|>"])[0]
    eoti = tokenizer.tokens2ids(["<|eot_id|>"])[0]
    special_tokens = {}
    special_tokens['<|begin_of_text|>'] = bos
    special_tokens['<|end_of_text|>'] = eos
    special_tokens['<|eot_id|>'] = eoti
    special_tokens['<|start_header_id|>'] = shi
    special_tokens['<|end_header_id|>'] = ehi
    return special_tokens


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
            name = name.replace('attention_norm.weight', 'norm1.weight')
            # att weight
            name = name.replace('.attention.wq.weight',
                                '.self_attn.linear_q.weight')
            name = name.replace('.attention.wk.weight',
                                '.self_attn.linear_k.weight')
            name = name.replace('.attention.wv.weight',
                                '.self_attn.linear_v.weight')
            # att out dim
            name = name.replace('attention.wo', 'self_attn.linear_out')
        else:
            # mlp
            name = name.replace('feed_forward.w1', 'feed_forward.gate')
            name = name.replace('feed_forward.w3', 'feed_forward.w_1')
            name = name.replace('feed_forward.w2', 'feed_forward.w_2')

            # before mlp ln: (rms norm)
            name = name.replace('ffn_norm', 'norm2')
        wenet_state_dict[name] = conformer_state_dict[old_name]
    # final norm weight
    wenet_state_dict['decoder.final_norm.weight'] = conformer_state_dict[
        'norm.weight']
    print("Saving {} ckpt to {}...".format(config.dtype,
                                           wenet_state_dict_path))
    torch.save(wenet_state_dict, wenet_state_dict_path)
    print(
        "DONE\n===================- End CKPT Conversion ====================\n"
    )
