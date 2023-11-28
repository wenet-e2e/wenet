import pytest

from wenet.text.wenet_tokenizer import WenetTokenizer


@pytest.mark.parametrize("symbol_table_path", [
    "test/resources/librispeech.words.txt",
    "test/resources/aishell2.words.txt",
])
def test_tokenize(symbol_table_path):
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
    if symbol_table_path == "test/resources/librispeech.words.txt":
        bpe_model = "test/resources/librispeech.train_960_unigram5000.bpemodel"
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
            "tokens":
            ['▁LIFE', '▁IS', '▁SHORT', '▁I', '▁USE', '▁WE', 'NE', 'T'],
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
            "label":
            [3, 4999, 2048, 1248, 2960, 2332, 4366, 4072, 3844, 1, 2901]
        }, {
            "tokens":
            ['▁', 'Z', 'HEN', 'DO', 'NG', '▁IS', 'T', '▁SO', '▁SCH', 'Ö', 'N'],
            "label":
            [3, 4999, 2048, 1248, 2960, 2332, 4366, 4072, 3844, 1, 2901]
        }, {
            "tokens": ['▁IT', "'", 'S', '▁O', 'KA', 'Y'],
            "label": [2344, 2, 3790, 3010, 2418, 4979]
        }]
    else:
        bpe_model = None
        refs = [{
            "tokens": ['震', '东', '好', '帅'],
            "label": [4932, 80, 1059, 1375]
        }, {
            "tokens": ['吴', '迪', '也', '好', '帅'],
            "label": [656, 4540, 117, 1059, 1375]
        }, {
            "tokens": [
                'b', 'i', 'n', 'b', 'i', 'n', '▁', 'i', 's', '▁', 'a', 'l',
                's', 'o', '▁', 'h', 'a', 'n', 'd', 's', 'o', 'm', 'e'
            ],
            "label": [
                9, 23, 33, 9, 23, 33, 1, 23, 43, 1, 7, 29, 43, 35, 1, 21, 7,
                33, 13, 43, 35, 31, 15
            ]
        }, {
            "tokens": [
                'l', 'i', 'f', 'e', '▁', 'i', 's', '▁', 's', 'h', 'o', 'r',
                't', '▁', 'i', '▁', 'u', 's', 'e', '▁', 'w', 'e', 'n', 'e', 't'
            ],
            "label": [
                29, 23, 17, 15, 1, 23, 43, 1, 43, 21, 35, 41, 46, 1, 23, 1, 48,
                43, 15, 1, 52, 15, 33, 15, 46
            ]
        }, {
            "tokens": [
                '超', '哥', '▁', 'i', 's', '▁', 't', 'h', 'e', '▁', 'm', 'o',
                's', 't', '▁', 'h', 'a', 'n', 'd', 's', 'o', 'm', 'e', '▁', '吧'
            ],
            "label": [
                4395, 736, 1, 23, 43, 1, 46, 21, 15, 1, 31, 35, 43, 46, 1, 21,
                7, 33, 13, 43, 35, 31, 15, 1, 647
            ]
        }, {
            "tokens": [
                '人', '生', '苦', '短', 'i', '▁', 'u', 's', 'e', '▁', 'w', 'e',
                'n', 'e', 't'
            ],
            "label":
            [155, 2980, 3833, 3178, 23, 1, 48, 43, 15, 1, 52, 15, 33, 15, 46]
        }, {
            "tokens": [
                '人', '生', '苦', '短', 'I', '▁', 'U', 'S', 'E', '▁', 'W', 'E',
                'N', 'E', 'T'
            ],
            "label":
            [155, 2980, 3833, 3178, 24, 1, 49, 44, 16, 1, 53, 16, 34, 16, 47]
        }, {
            "tokens": [
                'z', 'h', 'e', 'n', 'd', 'o', 'n', 'g', '▁', 'i', 's', 't',
                '▁', 's', 'o', '▁', 's', 'c', 'h', 'ö', 'n'
            ],
            "label": [
                58, 21, 15, 33, 13, 35, 33, 19, 1, 23, 43, 46, 1, 43, 35, 1,
                43, 11, 21, 1, 33
            ]
        }, {
            "tokens": [
                'z', 'h', 'e', 'n', 'd', 'o', 'n', 'g', '▁', 'i', 's', 't',
                '▁', 's', 'o', '▁', 's', 'c', 'h', 'ö', 'n'
            ],
            "label": [
                58, 21, 15, 33, 13, 35, 33, 19, 1, 23, 43, 46, 1, 43, 35, 1,
                43, 11, 21, 1, 33
            ]
        }, {
            "tokens": ['I', 't', "'", 's', '▁', 'o', 'k', 'a', 'y'],
            "label": [24, 46, 2, 43, 1, 35, 27, 7, 56]
        }]
    tokenizer = WenetTokenizer(symbol_table=symbol_table_path,
                               bpe_model=bpe_model,
                               split_with_space=False)
    results = []
    for line in txts:
        tokens, label = tokenizer.tokenize(line)
        results.append({"tokens": tokens, "label": label})

    for (hyp, ref) in zip(results, refs):
        print(hyp["tokens"], ref["tokens"])
        assert (len(hyp["tokens"]) == len(ref["tokens"]))
        assert (all(h == r for h, r in zip(hyp["tokens"], ref["tokens"])))
        assert (len(hyp["label"]) == len(ref["label"]))
        assert (all(h == r for h, r in zip(hyp["label"], ref["label"])))


@pytest.mark.parametrize("use_pbe_model", [True, False])
def test_non_lang_symbol_tokenize(use_pbe_model):
    data = ["我是{NOISE}"]
    symbol_table = {"我": 1, "是": 2, "{NOISE}": 3}
    bpe_model = None
    non_lang_syms = ["{NOISE}"]
    expected = ["我", "是", "{NOISE}"]

    if use_pbe_model:
        bpe_model = "test/resources/librispeech.train_960_unigram5000.bpemodel"

    tokenizer = WenetTokenizer(symbol_table,
                               bpe_model=bpe_model,
                               non_lang_syms=non_lang_syms)
    for line in data:
        tokens, _ = tokenizer.tokenize(line)
        assert (all(h == r for h, r in zip(tokens, expected)))


@pytest.mark.parametrize("symbol_table_path", [
    "test/resources/librispeech.words.txt",
    "test/resources/aishell2.words.txt",
])
def test_vocab_size(symbol_table_path):
    tokenizer = WenetTokenizer(symbol_table_path)
    if symbol_table_path == "test/resources/librispeech.words.txt":
        assert tokenizer.vocab_size() == 5002
    else:
        assert tokenizer.vocab_size() == 5235
