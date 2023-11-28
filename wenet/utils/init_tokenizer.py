from wenet.text.base_tokenizer import BaseTokenizer
from wenet.text.bpe_tokenizer import BpeTokenizer
from wenet.text.char_tokenizer import CharTokenizer
from wenet.text.whisper_tokenizer import WhisperTokenizer


def init_tokenizer(configs,
                   symbol_table,
                   bpe_model=None,
                   non_lang_syms=None) -> BaseTokenizer:
    # TODO:
    # 1 huggface tokenizer
    # 2 paraformer tokenizer

    if configs.get("whisper", False):
        tokenizer = WhisperTokenizer(
            multilingual=configs['whisper_conf']['is_multilingual'],
            num_languages=configs['whisper_conf']['num_languages'])
    elif bpe_model is None:
        tokenizer = CharTokenizer(symbol_table,
                                  non_lang_syms,
                                  split_with_space=configs.get(
                                      'split_with_space', False))
    else:
        tokenizer = BpeTokenizer(bpe_model,
                                 symbol_table,
                                 split_with_space=configs.get(
                                     'split_with_space', False))

    return tokenizer
