from functools import partial
from multiprocessing import Pool
from wenet.text.base_tokenizer import BaseTokenizer

from wenet.text.bpe_tokenizer import BpeTokenizer
from wenet.text.hugging_face_tokenizer import HuggingFaceTokenizer
from wenet.text.whisper_tokenizer import WhisperTokenizer


def consistency(tokenizer: BaseTokenizer, line: str) -> str:
    return tokenizer.detokenize(tokenizer.tokenize(line)[1])[0]


def test_whisper_tokenzier_parallel():

    inputs = ["it's ok", "wenet is simple", "test for new io"]
    tokenizer = WhisperTokenizer(False)

    partial_tokenize = partial(consistency, tokenizer)
    with Pool(processes=len(inputs)) as pool:
        results = pool.map(partial_tokenize, inputs)

    inputs.sort()
    results.sort()

    assert all(h == r for (h, r) in zip(results, inputs))


def test_whisper_tokenzier_parallel_after_property():

    inputs = ["it's ok", "wenet is simple", "test for new io"]
    tokenizer = WhisperTokenizer(False)

    _ = tokenizer.vocab_size
    _ = tokenizer.symbol_table
    partial_tokenize = partial(consistency, tokenizer)
    with Pool(processes=len(inputs)) as pool:
        results = pool.map(partial_tokenize, inputs)

    inputs.sort()
    results.sort()

    assert all(h == r for (h, r) in zip(results, inputs))


def test_bpe_tokenzier_parallel():

    symbol_table_path = "test/resources/librispeech.words.txt"
    bpe_model = "test/resources/librispeech.train_960_unigram5000.bpemodel"

    inputs = ["WENT IS SIMPLE", "GOOD"]
    tokenizer = BpeTokenizer(bpe_model, symbol_table_path)
    partial_tokenize = partial(consistency, tokenizer)
    with Pool(processes=len(inputs)) as pool:
        results = pool.map(partial_tokenize, inputs)

    inputs.sort()
    results.sort()

    assert all(h == r for (h, r) in zip(results, inputs))


def test_bpe_tokenizer_parallel_after_property():
    symbol_table_path = "test/resources/librispeech.words.txt"
    bpe_model = "test/resources/librispeech.train_960_unigram5000.bpemodel"

    inputs = ["WENT IS SIMPLE", "GOOD"]
    tokenizer = BpeTokenizer(bpe_model, symbol_table_path)
    _ = tokenizer.vocab_size
    _ = tokenizer.symbol_table

    partial_tokenize = partial(consistency, tokenizer)
    with Pool(processes=len(inputs)) as pool:
        results = pool.map(partial_tokenize, inputs)

    inputs.sort()
    results.sort()

    assert all(h == r for (h, r) in zip(results, inputs))


def test_hugging_face_tokenizer():
    tokenizer = HuggingFaceTokenizer("bert-base-cased")

    _ = tokenizer.vocab_size
    _ = tokenizer.symbol_table

    inputs = ["wenet is simple", "good"]
    partial_tokenize = partial(consistency, tokenizer)
    with Pool(processes=len(inputs)) as pool:
        results = pool.map(partial_tokenize, inputs)

    inputs.sort()
    results.sort()

    assert all(h == r for (h, r) in zip(results, inputs))
