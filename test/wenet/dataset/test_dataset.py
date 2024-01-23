import pytest
import torch
from wenet.dataset.datapipes import (WenetRawDatasetSource,
                                     WenetTarShardDatasetSource)
from wenet.dataset.processor_v2 import compute_fbank, decode_wav, parse_json


@pytest.mark.parametrize("data_list", [
    "test/resources/dataset/data.list",
])
def test_WenetRawDatasetSource(data_list):

    dataset = WenetRawDatasetSource(data_list)
    expected = []
    with open(data_list, 'r') as fin:
        for line in fin:
            line = line.strip('\n')
            expected.append({"file_name": data_list, "line": line})
    result = []
    for elem in dataset:
        result.append(elem)
    print(result)

    assert len(result) == len(expected)
    for (i, elem) in enumerate(result):
        for key, value in elem.items():
            assert key in expected[i].keys()
            assert value == expected[i][key]


@pytest.mark.parametrize("data_list", [(
    "test/resources/dataset/data.list",
    "test/resources/dataset/data.shards.list",
)])
def test_dataset_consistently(data_list):
    raw_list, tar_list = data_list
    raw_dataset = WenetRawDatasetSource(raw_list)
    raw_dataset = raw_dataset.map(parse_json)
    raw_dataset = raw_dataset.map(decode_wav)
    raw_dataset = raw_dataset.map(compute_fbank)
    raw_results = []
    for d in raw_dataset:
        raw_results.append(d)

    keys = ["key", "txt", "file_name", "wav", "sample_rate", "feat"]
    for r in raw_results:
        assert set(r.keys()) == set(keys)
    tar_dataset = WenetTarShardDatasetSource(tar_list)
    tar_dataset = tar_dataset.map(decode_wav)
    tar_dataset = tar_dataset.map(compute_fbank)
    tar_results = []
    for d in tar_dataset:
        tar_results.append(d)
    keys.append('tar_file_name')
    for r in tar_results:
        assert set(r.keys()) == set(keys)

    assert len(tar_results) == len(raw_results)
    sorted(tar_results, key=lambda elem: elem['key'])
    sorted(raw_results, key=lambda elem: elem['key'])
    same_keys = ["txt", "wav", "sample_rate", "feat"]
    for (i, tar_result) in enumerate(tar_results):
        for k in same_keys:
            if isinstance(tar_result[k], torch.Tensor):
                assert isinstance(raw_results[i][k], torch.Tensor)
                assert torch.allclose(tar_result[k], raw_results[i][k])
            else:
                assert tar_result[k] == raw_results[i][k]
