import json
from torch.utils.data import DataLoader
from wenet.dataset.datapipes import WenetRawDatasetSource, WenetShardDatasetSource
from wenet.dataset.processor_v2 import decode_wav


def parse_json(elem):
    line = elem['line']
    obj = json.loads(line)
    obj['file_name'] = elem['file_name']
    return dict(obj)


if __name__ == '__main__':

    dataset = WenetShardDatasetSource(
        'test/resources/dataset/data.shards.list')
    dataset = dataset.map(decode_wav)

    dataloader = DataLoader(dataset, num_workers=4, persistent_workers=True)
    for d in dataloader:
        print(d.keys())

    dataset = WenetRawDatasetSource("test/resources/dataset/data.list")

    dataset = dataset.map(parse_json)
    dataset = dataset.map(decode_wav)
    for d in dataset:
        print(d.keys())
