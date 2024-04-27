import argparse

import os

import torch

from wenet.LLM.script.config import (convert_to_wenet_yaml,
                                     gemma_config_for_2b, gemma_config_for_7b,
                                     llama3_config_for_70b,
                                     llama3_config_for_8b)
from wenet.LLM.script.gemma_config import (convert_to_wenet_state_dict as
                                           gemma_convert_ckpt_fn,
                                           gemma_special_tokens)
from wenet.LLM.script.llama3_config import (convert_to_wenet_state_dict as
                                            llama3_convert_ckpt_fn,
                                            llama3_special_tokens)


def get_args():
    parser = argparse.ArgumentParser(description='load and convert llm ckpt')
    parser.add_argument('--ckpt',
                        required=True,
                        help='llama3: https://llama.meta.com/llama-downloads/ \
         \ngemma: https://www.kaggle.com/models/google/gemma/frameworks/pyTorch'
                        )
    parser.add_argument('--model_size', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--output_dir',
                        default='.',
                        help='output file in wenet\'s style')
    args = parser.parse_args()
    return args


MODEL = {
    "gemma": {
        "2b": (gemma_config_for_2b(), 'google/gemma-2b'),
        "7b": (gemma_config_for_7b(), 'google/gemma-7b'),
        "ckpt_fn": gemma_convert_ckpt_fn,
        'tie_word_embeding': True,
        'special_tokens_fn': gemma_special_tokens,
    },
    "llama3": {
        "8b": (llama3_config_for_8b(), 'meta-llama/Meta-Llama-3-8B'),
        "70b": (llama3_config_for_70b(), 'meta-llama/Meta-Llama-3-70B'),
        "fn": llama3_convert_ckpt_fn,
        'tie_word_embeding': False,
        'special_tokens_fn': llama3_special_tokens,
    },
}


def main():
    args = get_args()
    args.jit = False
    model_size = args.model_size
    model_name = args.model_name
    assert model_name in MODEL.keys()
    all(model_size in size.keys() for size in MODEL.values())
    config = MODEL[model_name][model_size][0]
    args.tokenizer = MODEL[model_name][model_size][1]

    os.makedirs(args.output_dir, exist_ok=True)

    checkpoint = torch.load(args.ckpt, map_location="cpu")
    if model_name == 'gemma':
        checkpoint = checkpoint["model_state_dict"]
    wenet_ckpt_path = os.path.join(args.output_dir,
                                   'wenet_' + os.path.basename(args.ckpt))
    wenet_ckpt_path = os.path.splitext(wenet_ckpt_path)[0] + ".pt"
    convert_fn = MODEL[model_name]['ckpt_fn']
    convert_fn(checkpoint, wenet_ckpt_path, config)

    wenet_yaml_path = os.path.join(args.output_dir, 'train.yaml')
    convert_to_wenet_yaml(
        config,
        wenet_yaml_path,
        args.tokenizer,
        template=model_name,
        tie_word_embedding=MODEL[model_name]['tie_word_embeding'],
        special_tokens=MODEL[model_name]['special_tokens_fn'](args.tokenizer,
                                                              config))


if __name__ == '__main__':
    main()
