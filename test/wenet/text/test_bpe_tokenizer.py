import pytest
from wenet.text.bpe_tokenizer import BpeTokenizer


@pytest.fixture(params=[[
    "test/resources/librispeech.words.txt",
    "test/resources/librispeech.train_960_unigram5000.bpemodel"
]])
def bpe_tokenizer(request):
    symbol_table, bpe_model = request.param
    return BpeTokenizer(bpe_model, symbol_table)


def test_tokenize(bpe_tokenizer):
    tokenizer = bpe_tokenizer
    txts = [
        "震东好帅",
        " 吴迪也好帅 ",
        "binbin is also handsome",
        " life is short i use wenet ",
        "超哥 is the most handsome 吧",
        " 人生苦短i use wenet ",
        "人生苦短I USE WENET",
        "zhendong ist so schön",
        " zhendong ist so schön ",
        "It's okay",
    ]
    refs = [{
        "tokens": ['震', '东', '好', '帅'],
        "label": [1, 1, 1, 1]
    }, {
        "tokens": ['吴', '迪', '也', '好', '帅'],
        "label": [1, 1, 1, 1, 1]
    }, {
        "tokens": ['▁B', 'IN', 'B', 'IN', '▁IS', '▁ALSO', "▁HANDSOME"],
        "label": [347, 2216, 346, 2216, 2332, 143, 1990]
    }, {
        "tokens": ['▁LIFE', '▁IS', '▁SHORT', '▁I', '▁USE', '▁WE', 'NE', 'T'],
        "label": [2568, 2332, 3968, 2152, 4699, 4833, 2926, 4366]
    }, {
        "tokens": ['超', '哥', '▁IS', '▁THE', '▁MOST', '▁HANDSOME', '吧'],
        "label": [1, 1, 2332, 4435, 2860, 1990, 1]
    }, {
        "tokens": ['人', '生', '苦', '短', '▁I', '▁USE', '▁WE', 'NE', 'T'],
        "label": [1, 1, 1, 1, 2152, 4699, 4833, 2926, 4366]
    }, {
        "tokens": ['人', '生', '苦', '短', '▁I', '▁USE', '▁WE', 'NE', 'T'],
        "label": [1, 1, 1, 1, 2152, 4699, 4833, 2926, 4366]
    }, {
        "tokens":
        ['▁', 'Z', 'HEN', 'DO', 'NG', '▁IS', 'T', '▁SO', '▁SCH', 'Ö', 'N'],
        "label": [3, 4999, 2048, 1248, 2960, 2332, 4366, 4072, 3844, 1, 2901]
    }, {
        "tokens":
        ['▁', 'Z', 'HEN', 'DO', 'NG', '▁IS', 'T', '▁SO', '▁SCH', 'Ö', 'N'],
        "label": [3, 4999, 2048, 1248, 2960, 2332, 4366, 4072, 3844, 1, 2901]
    }, {
        "tokens": ['▁IT', "'", 'S', '▁O', 'KA', 'Y'],
        "label": [2344, 2, 3790, 3010, 2418, 4979]
    }]

    results = []
    for line in txts:
        result = tokenizer.tokenize(line)
        results.append(result)

    for (hyp, ref) in zip(results, refs):
        for module in hyp["tokens"].keys():
            assert (len(hyp["tokens"][module]) == len(ref["tokens"]))
            assert (all(h == r
                        for h, r in zip(hyp["tokens"][module], ref["tokens"])))
            assert (len(hyp["label"][module]) == len(ref["label"]))
            assert (all(h == r
                        for h, r in zip(hyp["label"][module], ref["label"])))


def test_detokenize(bpe_tokenizer):
    tokenizer = bpe_tokenizer
    # TODO(Mddct): more unit test
    ids = [2344, 2, 3790, 3010, 2418, 4979]
    expected = {
        'txt': "IT'S OKAY",
        "tokens": ['▁IT', "'", 'S', '▁O', 'KA', 'Y']
    }
    result = tokenizer.detokenize(ids)
    for module in result["tokens"].keys():
        assert result["text"][module] == expected['txt']
        assert (all(
            h == r
            for h, r in zip(result["tokens"][module], expected['tokens'])))


def test_vocab_size(bpe_tokenizer):
    assert bpe_tokenizer.vocab_size() == 5002


def test_consistency(bpe_tokenizer):
    text = "WENET IS GREAT"
    assert text == bpe_tokenizer.tokens2text(bpe_tokenizer.text2tokens(text))
    result = bpe_tokenizer.tokenize(text)["label"]
    for module in result.keys():
        assert text == bpe_tokenizer.detokenize(result[module])["text"][module]
