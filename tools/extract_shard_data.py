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

import os
import torch
import tarfile
import argparse
import torchaudio
import soundfile
import logging
from tqdm import tqdm
from urllib.parse import urlparse
from subprocess import Popen, PIPE
from torch.utils.data import IterableDataset, DataLoader

AUDIO_FORMAT_SETS = set(["flac", "mp3", "m4a", "ogg", "opus", "wav", "wma"])


class AudioIterableDataset(IterableDataset):
    def __init__(self, file_list, flag):
        self.file_list = file_list
        self.flag = flag

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
                        if postfix == "txt" and self.flag == "text":
                            example["txt"] = file_obj.read().decode("utf8").strip()
                        elif postfix in AUDIO_FORMAT_SETS and self.flag == "duration":
                            # only need duration
                            info = soundfile.info(file_obj)
                            example["duration"] = info.duration
                        elif postfix in AUDIO_FORMAT_SETS and self.flag == "audio":
                            waveform, sample_rate = torchaudio.load(file_obj)
                            example["wav"] = waveform
                            example["sample_rate"] = sample_rate
                        elif self.flag == "content":
                            example[postfix] = file_obj.read()
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
    parser = argparse.ArgumentParser(
        description="Extract text or duration from shard data"
    )
    parser.add_argument("in_file", type=str, help="Input file list")
    parser.add_argument("out_file", type=str, help="Output file")
    parser.add_argument(
        "flag",
        type=str,
        choices=["text", "duration", "audio", "content"],
        help="Mode: text or duration",
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of worker threads"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")

    args = parser.parse_args()

    dataset = AudioIterableDataset(args.in_file, args.flag)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=lambda x: x,
    )

    total_duration = 0.0

    if args.flag in ["text", "duration"]:
        f = open(args.out_file, "w")
    else:
        os.makedirs(args.out_file, exist_ok=True)

    with tqdm() as pbar:
        for batch in dataloader:
            for result in batch:
                key = result["key"]
                if args.flag == "text":
                    text = result["txt"]
                    print(f"{key} {text}", file=f)
                elif args.flag == "duration":
                    duration = result["duration"]
                    total_duration += duration
                    print(f"{key} {duration}", file=f)
                elif args.flag == "audio":
                    sample_rate = result["sample_rate"]
                    wav = result["wav"]
                    duration = wav.shape[-1] / sample_rate
                    total_duration += duration
                    torchaudio.save(f"{args.out_file}/{key}.wav", wav, sample_rate)
                elif args.flag == "content":
                    for postfix, content in result.items():
                        if postfix != "key":
                            out_file = f"{args.out_file}/{key}.{postfix}"
                            with open(out_file, "wb") as f1:
                                f1.write(content)
            pbar.update()

    if args.flag in ["text", "duration"]:
        f.close()

    print(
        f"Mode: {args.flag} total duration: {total_duration/3600} h "
        f"saved to: {args.out_file}"
    )


if __name__ == "__main__":
    main()
