import pytest
from wenet.text.char_tokenizer import CharTokenizer


@pytest.fixture(params=["test/resources/aishell2.words.txt"])
def char_tokenizer(request):
    symbol_table = request.param
    return CharTokenizer(symbol_table)


def test_tokenize(char_tokenizer):
    tokenizer = char_tokenizer
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
        "label": [4932, 80, 1059, 1375]
    }, {
        "tokens": ['吴', '迪', '也', '好', '帅'],
        "label": [656, 4540, 117, 1059, 1375]
    }, {
        "tokens": [
            'b', 'i', 'n', 'b', 'i', 'n', '▁', 'i', 's', '▁', 'a', 'l', 's',
            'o', '▁', 'h', 'a', 'n', 'd', 's', 'o', 'm', 'e'
        ],
        "label": [
            9, 23, 33, 9, 23, 33, 1, 23, 43, 1, 7, 29, 43, 35, 1, 21, 7, 33,
            13, 43, 35, 31, 15
        ]
    }, {
        "tokens": [
            'l', 'i', 'f', 'e', '▁', 'i', 's', '▁', 's', 'h', 'o', 'r', 't',
            '▁', 'i', '▁', 'u', 's', 'e', '▁', 'w', 'e', 'n', 'e', 't'
        ],
        "label": [
            29, 23, 17, 15, 1, 23, 43, 1, 43, 21, 35, 41, 46, 1, 23, 1, 48, 43,
            15, 1, 52, 15, 33, 15, 46
        ]
    }, {
        "tokens": [
            '超', '哥', '▁', 'i', 's', '▁', 't', 'h', 'e', '▁', 'm', 'o', 's',
            't', '▁', 'h', 'a', 'n', 'd', 's', 'o', 'm', 'e', '▁', '吧'
        ],
        "label": [
            4395, 736, 1, 23, 43, 1, 46, 21, 15, 1, 31, 35, 43, 46, 1, 21, 7,
            33, 13, 43, 35, 31, 15, 1, 647
        ]
    }, {
        "tokens": [
            '人', '生', '苦', '短', 'i', '▁', 'u', 's', 'e', '▁', 'w', 'e', 'n',
            'e', 't'
        ],
        "label":
        [155, 2980, 3833, 3178, 23, 1, 48, 43, 15, 1, 52, 15, 33, 15, 46]
    }, {
        "tokens": [
            '人', '生', '苦', '短', 'I', '▁', 'U', 'S', 'E', '▁', 'W', 'E', 'N',
            'E', 'T'
        ],
        "label":
        [155, 2980, 3833, 3178, 24, 1, 49, 44, 16, 1, 53, 16, 34, 16, 47]
    }, {
        "tokens": [
            'z', 'h', 'e', 'n', 'd', 'o', 'n', 'g', '▁', 'i', 's', 't', '▁',
            's', 'o', '▁', 's', 'c', 'h', 'ö', 'n'
        ],
        "label": [
            58, 21, 15, 33, 13, 35, 33, 19, 1, 23, 43, 46, 1, 43, 35, 1, 43,
            11, 21, 1, 33
        ]
    }, {
        "tokens": [
            'z', 'h', 'e', 'n', 'd', 'o', 'n', 'g', '▁', 'i', 's', 't', '▁',
            's', 'o', '▁', 's', 'c', 'h', 'ö', 'n'
        ],
        "label": [
            58, 21, 15, 33, 13, 35, 33, 19, 1, 23, 43, 46, 1, 43, 35, 1, 43,
            11, 21, 1, 33
        ]
    }, {
        "tokens": ['I', 't', "'", 's', '▁', 'o', 'k', 'a', 'y'],
        "label": [24, 46, 2, 43, 1, 35, 27, 7, 56]
    }]
    results = []
    for line in txts:
        tokens, label = tokenizer.tokenize(line)
        results.append({"tokens": tokens, "label": label})

    for (hyp, ref) in zip(results, refs):
        assert (len(hyp["tokens"]) == len(ref["tokens"]))
        assert (all(h == r for h, r in zip(hyp["tokens"], ref["tokens"])))
        assert (len(hyp["label"]) == len(ref["label"]))
        assert (all(h == r for h, r in zip(hyp["label"], ref["label"])))


def test_detokenize(char_tokenizer):
    tokenizer = char_tokenizer
    idss = [
        [4932, 80, 1059, 1375],
        [656, 4540, 117, 1059, 1375],
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


def test_vocab_size(char_tokenizer):
    assert char_tokenizer.vocab_size() == 5235


def test_consistency(char_tokenizer):
    text = "大家都好帅"

    assert text == char_tokenizer.tokens2text(char_tokenizer.text2tokens(text))
    assert text == char_tokenizer.detokenize(
        char_tokenizer.tokenize(text)[1])[0]
