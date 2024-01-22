from torch.utils.data import DataLoader
from wenet.dataset.datapipes import WenetShardDatasetSource

if __name__ == '__main__':

    dataset = WenetShardDatasetSource(
        'test/resources/dataset/data.shards.list')
    dataloader = DataLoader(dataset, num_workers=4, persistent_workers=True)
    for d in dataloader:
        print(d['key'])
