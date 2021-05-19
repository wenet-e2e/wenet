#coding=utf-8

import os,sys
import requests
from multiprocessing import freeze_support,Pool
import time
import json
import random
import numpy as np

import signal
def exit_func(signum, frame):
    print('exit function')
    os._exit(0)



def worker(index, filename, host):
    try:
        print('{} process starting'.format(index), file=sys.stderr)
        audio = open(filename, 'rb')
        audio_name = os.path.basename(filename)
        #  files = {'audio': (audio_name, audio.read(), 'audio/wav')}
        files = {'': (audio_name, audio.read(), 'audio')}
        audio.close()
        
        data = {
            'sample_rate': 16000,
        }

        start = time.time()
        r = requests.post('{}/asr'.format(host.rstrip('/')), data=data, files=files, verify=False, timeout=10)
        #  r = requests.post('http://{}/asr'.format(host))
        elapsed = time.time() - start
        print('elapsed time:{}s'.format(elapsed), file=sys.stderr)

        ret = json.loads(r.text)
        # print(ret, file=sys.stderr)
        wave_time = float(ret['duration'])
        if wave_time != 0:
            rtf = elapsed / wave_time
        else:
            raise Exception('Error status:{} message:{}'.format(ret['ret_code'], ret['ret_msg']))
        if int(ret['ret_code']) != 1:
            raise Exception('Error status:{} message:{}'.format(ret['ret_code'], ret['ret_msg']))
        print('filename: {}, index: {} text: {}, duration: {}s\nrtf:{}'.format(filename, index, ret['result'], wave_time, rtf), file=sys.stderr)
        print('{} process finished'.format(index), file=sys.stderr)
        return rtf, os.path.basename(filename).rsplit('.', 1)[0], ret['result']
    except Exception as e:
        #  print('error response:{}'.format(r.text), file=sys.stderr)
        print('{} process, filename: {}, worker error:{}'.format(index, filename, str(e)), file=sys.stderr)
        return -1, "", ""


if __name__ == "__main__":
    signal.signal(signal.SIGINT, exit_func)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('audio_dir', type=str,  help="the input dir of audio")
    parser.add_argument('times', type=int, help='the times of test')
    parser.add_argument('host', type=str)
    parser.add_argument('--concurrent', type=int, dest="concurrent",
                        default=1,help='the number of concurrent. use 1 if the test is rtf')
    args = parser.parse_args()
    file_list = os.listdir(args.audio_dir)
    freeze_support()
    if args.concurrent < 20:
        pool = Pool(args.concurrent * 20)
    else:
        pool = Pool(args.concurrent * 5)
    start = time.time()
    #  print(file_list)
    file_list = [ x for x in file_list if x.endswith('wav') ]
    divide = len(file_list)
    #  print(file_list)
    result = []
    for i in range(args.times):
        filename = os.path.join(args.audio_dir, file_list[i % divide])
        # worker(i, filename, args.host)
        result.append(pool.apply_async(func=worker, args=(i, filename, args.host, )))
        if (i + 1) % args.concurrent == 0:
            print('time sleep 1s', file=sys.stderr)
            time.sleep(1)
    pool.close()
    pool.join()
    num_success = 0
    rtfs = []
    for i in result:
        try:
            items = i.get()
            print(items)
            rtf = items[0]
            if rtf > 0:
                rtfs.append(rtf)
                uttid = items[1]
                words = items[2]
                print('{} {}\n'.format(uttid, words))
                num_success += 1
        except:
            pass
    # print(len(rtfs))
    rtfs = np.asarray(rtfs, dtype=np.float32)
    print(rtfs)
    print('min rtf:', np.min(rtfs), file=sys.stderr)
    print('max rtf:', np.max(rtfs), file=sys.stderr)
    print('mean rtf:', np.mean(rtfs), file=sys.stderr)
    elapsed = time.time() - start
    print("total cost time: {}s".format(elapsed), file=sys.stderr)
    print('times: {}, success: {}, ratio: {}'.format(args.times, num_success, num_success / args.times), file=sys.stderr)
