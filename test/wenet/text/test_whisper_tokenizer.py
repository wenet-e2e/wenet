import pytest

from wenet.text.whisper_tokenizer import WhisperTokenizer


@pytest.fixture(params=[False])
def whisper_tokenizer(request):
    is_multilingual = request.param
    return WhisperTokenizer(is_multilingual)


def test_tokenize(whisper_tokenizer):

    tokenizer = whisper_tokenizer
    texts = ["life is short, i use wenet"]
    expected = [{
        "tokens": [
            "b'life'", "b'<space>is'", "b'<space>short'", "b','",
            "b'<space>i'", "b'<space>use'", "b'<space>w'", "b'en'", "b'et'"
        ],
        "ids": [6042, 318, 1790, 11, 1312, 779, 266, 268, 316],
    }]

    for i, text in enumerate(texts):
        result = tokenizer.tokenize(text)
        for module in result["tokens"].keys():
            assert len(result["label"][module]) == len(result["label"][module])
            assert (all((h == r for h, r in zip(result["tokens"][module], expected[i]["tokens"]))))
            assert (all((h == r for h, r in zip(result["label"][module], expected[i]["ids"]))))


def test_detokenize(whisper_tokenizer):
    tokenize = whisper_tokenizer

    inputs = [[6042, 318, 1790, 11, 1312, 779, 266, 268, 316]]
    expected = [{
        "tokens": [
            "b'life'", "b'<space>is'", "b'<space>short'", "b','",
            "b'<space>i'", "b'<space>use'", "b'<space>w'", "b'en'", "b'et'"
        ],
        'labels':
        "life is short, i use wenet",
    }]

    for i, input in enumerate(inputs):
        result = tokenize.detokenize(input)
        for module in result["tokens"].keys():
            assert len(result["tokens"][module]) == len(expected[i]["tokens"])
            assert result["text"][module] == expected[i]["labels"]
            assert all((h == r for h, r in zip(result["tokens"][module], expected[i]["tokens"])))


def test_consistency(whisper_tokenizer):
    text = "whisper powered by wenet, it's great"

    assert text == whisper_tokenizer.tokens2text(
        whisper_tokenizer.text2tokens(text))


def test_vocab_size(whisper_tokenizer):
    assert whisper_tokenizer.vocab_size(
    ) == whisper_tokenizer.tokenizer.encoding.n_vocab
