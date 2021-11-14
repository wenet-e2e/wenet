# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import multiprocessing
from multiprocessing import Pool

import argparse
import os
import tritonclient.grpc as grpcclient
from utils import cal_cer
from offline_client import SpeechClient
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        default='localhost:8001',
                        help='Inference server URL. Default is '
                        'localhost:8001.')
    parser.add_argument('--model_name',
                        required=False,
                        default='attention_rescoring',
                        help='the model to send request to')
    parser.add_argument('--wavscp',
                        type=str,
                        required=False,
                        default=None,
                        help='audio_id \t absolute_wav_path')
    parser.add_argument('--trans',
                        type=str,
                        required=False,
                        default=None,
                        help='audio_id \t text')
    parser.add_argument('--data_dir',
                        type=str,
                        required=False,
                        default=None,
                        help='data dir will be append to audio file if given')
    parser.add_argument(
        '--audio_file',
        type=str,
        required=False,
        default=None,
        help='single wav file'
    )

    FLAGS = parser.parse_args()

    # load data
    filenames = []
    transcripts = []
    if FLAGS.audio_file is not None:
        path = FLAGS.audio_file
        if FLAGS.data_dir:
            path = os.path.join(FLAGS.data_dir, path)
        if os.path.exists(path):
            filenames = [path]
    elif FLAGS.wavscp is not None:
        audio_data = {}
        with open(FLAGS.wavscp, "r", encoding="utf-8") as f:
            for line in f:
                aid, path = line.strip().split('\t')
                if FLAGS.data_dir:
                    path = os.path.join(FLAGS.data_dir, path)
                audio_data[aid] = {'path': path}
        with open(FLAGS.trans, "r", encoding="utf-8") as f:
            for line in f:
                aid, text = line.strip().split('\t')
                audio_data[aid]['text'] = text
        for key, value in audio_data.items():
            filenames.append(value['path'])
            transcripts.append(value['text'])

    num_workers = multiprocessing.cpu_count() // 2

    def single_job(client_files):
        with grpcclient.InferenceServerClient(url=FLAGS.url,
                                              verbose=FLAGS.verbose) as triton_client:
            protocol_client = grpcclient
            speech_client = SpeechClient(triton_client, FLAGS.model_name,
                                         protocol_client)
            idx, audio_files = client_files
            predictions = []
            for li in audio_files:
                result = speech_client.recognize(li, idx)
                print("Recognized {}:{}".format(li, result[0]))
                predictions += result
        return predictions

    # start to do inference
    # Group requests in batches
    predictions = []
    tasks = []
    splits = np.array_split(filenames, num_workers)

    for idx, per_split in enumerate(splits):
        cur_files = per_split.tolist()
        tasks.append((idx, cur_files))

    with Pool(processes=num_workers) as pool:
        predictions = pool.map(single_job, tasks)

    predictions = [item for sublist in predictions for item in sublist]
    if transcripts:
        cer = cal_cer(predictions, transcripts)
        print("CER is: {}".format(cer))
