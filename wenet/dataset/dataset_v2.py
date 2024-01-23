from torch.utils.data import DataLoader
from wenet.dataset.datapipes import (WenetRawDatasetSource,
                                     WenetTarShardDatasetSource)
from wenet.dataset.processor_v2 import (compute_fbank,
                                        compute_log_mel_spectrogram,
                                        decode_wav, parse_json)

if __name__ == '__main__':

    dataset = WenetTarShardDatasetSource(
        'test/resources/dataset/data.shards.list')
    # dataset = dataset.map(decode_wav)
    # dataset = dataset.map(compute_fbank)
    # dataset = dataset.map(compute_log_mel_spectrogram)
    dataloader = DataLoader(dataset,
                            num_workers=1,
                            persistent_workers=True,
                            batch_size=None)
    for d in dataloader:
        print(d['file_name'], d['tar_file_name'], d['txt'], len(d['wav']))
    for d in dataloader:
        print(d['file_name'], d['tar_file_name'], d['txt'], len(d['wav']))
    for d in dataloader:
        print(d['file_name'], d['tar_file_name'], d['txt'], len(d['wav']))

    dataset = dataset.map(decode_wav).map(compute_fbank)
    dataloader = DataLoader(dataset,
                            num_workers=1,
                            persistent_workers=True,
                            batch_size=None)
    for d in dataloader:
        print(d['file_name'], d['tar_file_name'], d['txt'], len(d['wav']),
              d['feat'].size())

    dataset = WenetRawDatasetSource("test/resources/dataset/data.list",
                                    prefetch=10)
    dataset = dataset.map(parse_json)
    dataset = dataset.map(decode_wav)
    dataset = dataset.map(compute_log_mel_spectrogram)
    dataloader = DataLoader(dataset,
                            num_workers=2,
                            persistent_workers=True,
                            batch_size=None)

    for d in dataloader:
        print(d.keys())

    for d in dataloader:
        print(d.keys())
