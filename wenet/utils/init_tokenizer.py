from wenet.text.base_tokenizer import BaseTokenizer
from wenet.text.wenet_tokenizer import WenetTokenizer
from wenet.text.whisper_tokenizer import WhisperTokenizer


def init_tokenizer(configs, args, non_lang_syms) -> BaseTokenizer:
    if configs.get("whisper", False):
        tokenizer = WhisperTokenizer(
            multilingual=configs['whisper_conf']['is_multilingual'],
            num_languages=configs['whisper_conf']['num_languages'])
    else:
        tokenizer = WenetTokenizer(args.symbol_table, args.bpe_model,
                                   non_lang_syms,
                                   configs.get('split_with_space', False))

    return tokenizer
