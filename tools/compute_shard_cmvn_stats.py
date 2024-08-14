#!/usr/bin/env python3

# Copyright (c) 2024 Timekettle Inc. (authors: Sirui Li)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import json
import yaml
import tarfile
import logging
import argparse

import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from torch.utils.data import IterableDataset, DataLoader
from urllib.parse import urlparse
from subprocess import Popen, PIPE

AUDIO_FORMAT_SETS = set(["flac", "mp3", "m4a", "ogg", "opus", "wav", "wma"])


class CollateFunc(object):
    """Collate function for AudioDataset"""

    def __init__(self, feat_dim, resample_rate):
        self.feat_dim = feat_dim
        self.resample_rate = resample_rate

    def __call__(self, batch):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            worker_id = worker_info.id
        else:
            worker_id = 0
        mean_stat = torch.zeros(self.feat_dim)
        var_stat = torch.zeros(self.feat_dim)
        number = 0
        batch_num = len(batch)
        for item in batch:
            try:
                waveform = item["wav"]
                sample_rate = item["sample_rate"]
                resample_rate = sample_rate
            except Exception as e:
                print(f"{item} read failed")
                continue
            waveform = waveform * (1 << 15)
            if self.resample_rate != 0 and self.resample_rate != sample_rate:
                resample_rate = self.resample_rate
                waveform = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=resample_rate
                )(waveform)

            mat = kaldi.fbank(
                waveform,
                num_mel_bins=self.feat_dim,
                dither=0.0,
                energy_floor=0.0,
                sample_frequency=resample_rate,
            )
            mean_stat += torch.sum(mat, axis=0)
            var_stat += torch.sum(torch.square(mat), axis=0)
            number += mat.shape[0]
        return number, mean_stat, var_stat, worker_id, batch_num


class AudioIterableDataset(IterableDataset):
    def __init__(self, file_list):
        self.file_list = file_list

    def parse_file_list(self):
        worker_info = torch.utils.data.get_worker_info()
        with open(self.file_list, "r") as f:
            parsed_files = [{"src": line.strip()} for line in f]

        if worker_info:
            # split workload
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            parsed_files = parsed_files[worker_id::num_workers]

        return parsed_files

    def url_opener(self, data):
        for sample in data:
            assert "src" in sample
            url = sample["src"]
            try:
                pr = urlparse(url)
                # local file
                if pr.scheme == "" or pr.scheme == "file":
                    stream = open(url, "rb")
                # network file, such as HTTP(HDFS/OSS/S3)/HTTPS/SCP
                else:
                    cmd = f"wget -q -O - {url}"
                    process = Popen(cmd, shell=True, stdout=PIPE)
                    sample.update(process=process)
                    stream = process.stdout
                sample.update(stream=stream)
                yield sample
            except Exception as ex:
                logging.warning("Failed to open {}".format(url))

    def tar_file_and_group(self, sample):
        assert "stream" in sample
        stream = None
        results = []
        try:
            stream = tarfile.open(fileobj=sample["stream"], mode="r:*")
            prev_prefix = None
            example = {}
            valid = True
            for tarinfo in stream:
                name = tarinfo.name
                pos = name.rfind(".")
                assert pos > 0
                prefix, postfix = name[:pos], name[pos + 1 :]
                if prev_prefix is not None and prefix != prev_prefix:
                    example["key"] = prev_prefix
                    if valid:
                        results.append(example)
                    example = {}
                    valid = True
                with stream.extractfile(tarinfo) as file_obj:
                    try:
                        if postfix in AUDIO_FORMAT_SETS:
                            waveform, sample_rate = torchaudio.load(file_obj)
                            example["wav"] = waveform
                            example["sample_rate"] = sample_rate
                    except Exception as ex:
                        valid = False
                        logging.warning("{} error to parse {}".format(ex, name))
                prev_prefix = prefix
            if prev_prefix is not None:
                example["key"] = prev_prefix
                results.append(example)
        except Exception as ex:
            logging.warning(
                "In tar_file_and_group: {} when processing {}".format(ex, sample["src"])
            )
        finally:
            if stream is not None:
                stream.close()
            if "process" in sample:
                sample["process"].communicate()
            sample["stream"].close()
        return results

    def __iter__(self):
        parsed_files = self.parse_file_list()
        for sample in self.url_opener(parsed_files):
            yield from self.tar_file_and_group(sample)


def main():
    parser = argparse.ArgumentParser(description="extract CMVN stats")
    parser.add_argument(
        "--num_workers",
        default=8,
        type=int,
        help="num of subprocess workers for processing",
    )
    parser.add_argument(
        "--batch_size", default=16, type=int, help="num of samples in a batch"
    )
    parser.add_argument("--train_config", default="", help="training yaml conf")
    parser.add_argument("--in_shard", default=None, help="shard data list file")
    parser.add_argument("--out_cmvn", default="global_cmvn", help="global cmvn file")
    parser.add_argument(
        "--log_interval",
        type=int,
        default=1000,
        help="Print log after every log_interval audios are processed.",
    )
    args = parser.parse_args()

    with open(args.train_config, "r") as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    feat_dim = (
        configs.get("dataset_conf", {}).get("fbank_conf", {}).get("num_mel_bins", 80)
    )
    resample_rate = (
        configs.get("dataset_conf", {}).get("resample_conf", {}).get("resample_rate", 0)
    )
    print(
        "compute cmvn using feat_dim: {} resample rate: {}".format(
            feat_dim, resample_rate
        )
    )
    collate_func = CollateFunc(feat_dim, resample_rate)
    dataset = AudioIterableDataset(args.in_shard)
    batch_size = args.batch_size

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_func,
    )

    with torch.no_grad():
        all_number = 0
        all_mean_stat = torch.zeros(feat_dim)
        all_var_stat = torch.zeros(feat_dim)
        wav_number = 0
        for i, batch in enumerate(data_loader):
            number, mean_stat, var_stat, worker_id, batch_num = batch
            all_mean_stat += mean_stat
            all_var_stat += var_stat
            all_number += number
            wav_number += batch_num

            if wav_number % args.log_interval == 0:
                print(
                    f"worker_id {worker_id} processed {wav_number} wavs "
                    f"{all_number} frames",
                    file=sys.stderr,
                    flush=True,
                )

    cmvn_info = {
        "mean_stat": list(all_mean_stat.tolist()),
        "var_stat": list(all_var_stat.tolist()),
        "frame_num": all_number,
    }

    with open(args.out_cmvn, "w") as fout:
        fout.write(json.dumps(cmvn_info))


if __name__ == "__main__":
    main()
