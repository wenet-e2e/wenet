import pytest

from wenet.utils.init_tokenizer import init_tokenizer


def test_init_whisper_tokenizer():
    # TODO(Mddct): add configs generator
    configs = {}
    configs['tokenizer'] = 'whisper'
    configs['tokenizer_conf'] = {}
    configs['tokenizer_conf']['symbol_table_path'] = None
    configs['tokenizer_conf']['is_multilingual'] = False
    configs['tokenizer_conf']['num_languages'] = 99

    tokenizer = init_tokenizer(configs)
    text = "whisper powered by wenet, it's great"

    assert text == tokenizer.tokens2text(tokenizer.text2tokens(text))


@pytest.mark.parametrize("symbol_table_path", [
    "test/resources/aishell2.words.txt",
])
def test_init_char_tokenizer(symbol_table_path):
    configs = {}
    configs['tokenizer'] = 'char'
    configs['tokenizer_conf'] = {}
    configs['tokenizer_conf']['symbol_table_path'] = symbol_table_path
    configs['tokenizer_conf']['non_lang_syms_path'] = None
    tokenizer = init_tokenizer(configs)

    text = "大家都好帅"
    assert text == tokenizer.tokens2text(tokenizer.text2tokens(text))


@pytest.mark.parametrize(
    "symbol_table_path, bpe_model",
    [("test/resources/librispeech.words.txt",
      "test/resources/librispeech.train_960_unigram5000.bpemodel")])
def test_init_bpe_tokenizer(symbol_table_path, bpe_model):

    configs = {}
    configs['tokenizer'] = 'bpe'
    configs['tokenizer_conf'] = {}
    configs['tokenizer_conf']['bpe_path'] = bpe_model
    configs['tokenizer_conf']['symbol_table_path'] = symbol_table_path
    configs['tokenizer_conf']['non_lang_syms_path'] = None
    tokenizer = init_tokenizer(configs)
    text = "WENET IT'S GREAT"

    assert text == tokenizer.tokens2text(tokenizer.text2tokens(text))


@pytest.mark.parametrize("symbol_table_path, seg_dict_path",
                         [("test/resources/paraformer.words.txt",
                           "test/resources/paraformer.seg_dict.txt")])
def test_init_paraformer_tokenizer(symbol_table_path, seg_dict_path):

    configs = {}
    configs['tokenizer'] = 'paraformer'
    configs['tokenizer_conf'] = {}
    configs['tokenizer_conf']['symbol_table_path'] = symbol_table_path
    configs['tokenizer_conf']['seg_dict_path'] = seg_dict_path
    tokenizer = init_tokenizer(configs)
    text = "paraformer powered by wenet,太棒了"

    assert text == tokenizer.tokens2text(tokenizer.tokenize(text)[0])
