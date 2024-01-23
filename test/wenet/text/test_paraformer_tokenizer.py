import pytest
from wenet.utils.init_tokenizer import init_tokenizer


@pytest.fixture(params=[[
    "test/resources/paraformer.words.txt",
    "test/resources/paraformer.seg_dict.txt"
]])
def paraformer_tokenizer(request):
    symbol_table_path, seg_dict = request.param
    configs = {}
    configs['model'] = 'paraformer'
    configs['tokenizer'] = 'paraformer'
    configs['tokenizer_conf'] = {}
    configs['tokenizer_conf']['symbol_table_path'] = symbol_table_path
    configs['tokenizer_conf']['seg_dict_path'] = seg_dict
    return init_tokenizer(configs)


def test_tokenize(paraformer_tokenizer):
    tokenizer = paraformer_tokenizer
    txts = ["震东好帅", " 吴迪也好帅 ", "星辰yyds", "kiss原则wenet looks     great"]
    expected = [
        {
            "tokens": ['震', '东', '好', '帅'],
            "label": [3987, 671, 4770, 6832],
        },
        {
            "tokens": ['吴', '迪', '也', '好', '帅'],
            "label": [854, 7543, 3567, 4770, 6832],
        },
        {
            "tokens": ['星', '辰', 'yyds'],
            "label": [7297, 1605, 8403],
        },
        {
            "tokens":
            ['ki@@', 'ss', '原', '则', 'wenet', 'loo@@', 'ks', 'great'],
            "label": [7154, 2010, 1895, 2289, 8403, 6946, 6609, 4683],
        },
    ]
    assert '<unk>' in tokenizer.symbol_table and tokenizer.symbol_table[
        tokenizer.unk] == 8403
    for (i, txt) in enumerate(txts):
        tokens, labels = tokenizer.tokenize(txt)
        assert len(tokens) == len(expected[i]['tokens'])
        assert len(labels) == len(expected[i]['label'])
        assert labels == expected[i]['label']
        assert all(h == r for (h, r) in zip(tokens, expected[i]['tokens']))


def test_detokenize(paraformer_tokenizer):
    tokenizer = paraformer_tokenizer
    idss = [
        [3987, 671, 4770, 6832],
        [854, 7543, 3567, 4770, 6832],
    ]

    refs = [{
        "txt": "震东好帅",
        "tokens": ['震', '东', '好', '帅'],
    }, {
        "txt": "吴迪也好帅",
        "tokens": ['吴', '迪', '也', '好', '帅'],
    }]
    results = []
    for ids in idss:
        txt, tokens = tokenizer.detokenize(ids)
        results.append({"tokens": tokens, "txt": txt})

    for (hyp, ref) in zip(results, refs):
        assert (len(hyp["tokens"]) == len(ref["tokens"]))
        assert (all(h == r for h, r in zip(hyp["tokens"], ref["tokens"])))
        assert len(hyp["txt"]) == len(ref["txt"])
        assert (all(h == r for h, r in zip(hyp["txt"], ref["txt"])))


def test_vocab_size(paraformer_tokenizer):
    assert paraformer_tokenizer.vocab_size() == 8403 + 1


def test_consistency(paraformer_tokenizer):
    text = "大家都好帅"

    assert text == paraformer_tokenizer.tokens2text(
        paraformer_tokenizer.text2tokens(text))
    assert text == paraformer_tokenizer.detokenize(
        paraformer_tokenizer.tokenize(text)[1])[0]

    text = "paraformer powered by wenet,太棒了"
    assert text == paraformer_tokenizer.tokens2text(
        paraformer_tokenizer.tokenize(text)[0])
