import pytest
from wenet.dataset.dataset_v2 import WenetRawDatasetSource


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
