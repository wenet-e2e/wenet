import pytest

from wenet.utils.init_tokenizer import init_tokenizer


def test_init_whisper_tokenizer():
    # TODO(Mddct): add configs generator
    configs = {}
    configs['whisper'] = True
    configs['whisper_conf'] = {}
    configs['whisper_conf']['is_multilingual'] = False
    configs['whisper_conf']['num_languages'] = 99

    tokenizer = init_tokenizer(configs, None)
    text = "whisper powered by wenet, it's great"

    assert text == tokenizer.tokens2text(tokenizer.text2tokens(text))


@pytest.mark.parametrize("symbol_table_path", [
    "test/resources/aishell2.words.txt",
])
def test_init_char_tokenizer(symbol_table_path):
    configs = {}
    tokenizer = init_tokenizer(configs, symbol_table_path)

    text = "大家都好帅"
    assert text == tokenizer.tokens2text(tokenizer.text2tokens(text))


@pytest.mark.parametrize(
    "symbol_table_path, bpe_model",
    [("test/resources/librispeech.words.txt",
      "test/resources/librispeech.train_960_unigram5000.bpemodel")])
def test_init_bpe_tokenizer(symbol_table_path, bpe_model):

    configs = {}
    tokenizer = init_tokenizer(configs, symbol_table_path, bpe_model)
    text = "WENET IT'S GREAT"

    assert text == tokenizer.tokens2text(tokenizer.text2tokens(text))
