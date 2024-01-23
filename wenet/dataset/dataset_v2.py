from torch.utils.data import DataLoader
from wenet.dataset.datapipes import (WenetRawDatasetSource,
                                     WenetTarShardDatasetSource)
from wenet.dataset.processor_v2 import (compute_fbank,
                                        compute_log_mel_spectrogram,
                                        decode_wav, parse_json)

if __name__ == '__main__':

    dataset = WenetTarShardDatasetSource(
        'test/resources/dataset/data.shards.list')
    dataset = dataset.map(decode_wav)
    dataset = dataset.map(compute_fbank)
    # dataset = dataset.map(compute_log_mel_spectrogram)
    dataloader = DataLoader(dataset, num_workers=4, persistent_workers=True)
    for d in dataloader:
        print(d['feat'])

    dataset = WenetRawDatasetSource("test/resources/dataset/data.list")

    dataset = dataset.map(parse_json)
    dataset = dataset.map(decode_wav)
    dataset = dataset.map(compute_log_mel_spectrogram)
    for d in dataset:
        print(d.keys())
