#!/usr/bin/env python3
# coding:utf-8

# Copyright (c) 2022 SDCI Co. Ltd (author: veelion)
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
import json
import time
import asyncio
import argparse
import websockets
import soundfile as sf
import statistics


WS_START = json.dumps({
    'signal': 'start',
    'nbest': 1,
    'continuous_decoding': False,
})
WS_END = json.dumps({
    'signal': 'end'
})


async def ws_rec(data, ws_uri):
    begin = time.time()
    conn = await websockets.connect(ws_uri, ping_timeout=200)
    # step 1: send start
    await conn.send(WS_START)
    ret = await conn.recv()
    # step 2: send audio data
    await conn.send(data)
    # step 3: send end
    await conn.send(WS_END)
    # step 4: receive result
    texts = []
    while 1:
        ret = await conn.recv()
        ret = json.loads(ret)
        if ret['type'] == 'final_result':
            nbest = json.loads(ret['nbest'])
            text = nbest[0]['sentence']
            texts.append(text)
        elif ret['type'] == 'speech_end':
            break
    # step 5: close
    try:
        await conn.close()
    except Exception as e:
        # this except has no effect, just log as debug
        # it seems the server does not send close info, maybe
        print(e)
    time_cost = time.time() - begin
    return {
        'text': ''.join(texts),
        'time': time_cost,
    }


def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '-u', '--ws_uri', required=True,
        help="websocket_server_main's uri, e.g. ws://127.0.0.1:10086")
    parser.add_argument(
        '-w', '--wav_scp', required=True,
        help='path to wav_scp_file')
    parser.add_argument(
        '-t', '--trans', required=True,
        help='path to trans_text_file of wavs')
    parser.add_argument(
        '-s', '--save_to', required=True,
        help='path to save transcription')
    parser.add_argument(
        '-n', '--num_concurrence', type=int, required=True,
        help='num of concurrence for query')
    args = parser.parse_args()
    return args


def print_result(info):
    length = max([len(k) for k in info])
    for k, v in info.items():
        print(f'\t{k: >{length}} : {v}')


async def main(args):
    wav_scp = []
    total_duration = 0
    with open(args.wav_scp) as f:
        for line in f:
            zz = line.strip().split()
            assert len(zz) == 2
            data, sr = sf.read(zz[1], dtype='int16')
            assert sr == 16000
            duration = (len(data)) / 16000
            total_duration += duration
            wav_scp.append((zz[0], data.tobytes()))
    print(f'{len(wav_scp) = }, {total_duration = }')

    tasks = []
    failed = 0
    texts = []
    request_times = []
    begin = time.time()
    for i, (_uttid, data) in enumerate(wav_scp):
        task = asyncio.create_task(ws_rec(data, args.ws_uri))
        tasks.append((_uttid, task))
        if len(tasks) < args.num_concurrence:
            continue
        print((f'{i=}, start {args.num_concurrence} '
               f'queries @ {time.strftime("%m-%d %H:%M:%S")}'))
        for uttid, task in tasks:
            result = await task
            texts.append(f'{uttid}\t{result["text"]}\n')
            request_times.append(result['time'])
        tasks = []
        print(f'\tdone @ {time.strftime("%m-%d %H:%M:%S")}')
    if tasks:
        for uttid, task in tasks:
            result = await task
            texts.append(f'{uttid}\t{result["text"]}\n')
            request_times.append(result['time'])
    request_time = time.time() - begin
    rtf = request_time / total_duration
    print('For all concurrence:')
    print_result({
        'failed': failed,
        'total_duration': total_duration,
        'request_time': request_time,
        'RTF': rtf,
    })
    print('For one request:')
    print_result({
        'mean': statistics.mean(request_times),
        'median': statistics.median(request_times),
        'max_time': max(request_times),
        'min_time': min(request_times),
    })
    with open(args.save_to, 'w', encoding='utf8') as fsave:
        fsave.write(''.join(texts))
    # caculate CER
    cmd = (f'python ../compute-wer.py --char=1 --v=1 '
           f'{args.trans} {args.save_to} > '
           f'{args.save_to}-test-{args.num_concurrence}.cer.txt')
    print(cmd)
    os.system(cmd)
    print('done')


if __name__ == '__main__':
    args = get_args()
    asyncio.run(main(args))
