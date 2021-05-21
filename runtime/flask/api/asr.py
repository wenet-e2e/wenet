# coding=utf-8

import json
import uuid
from flask import request
from api import server

import time
from config.config import ServerConfig
server_config = ServerConfig()
from interface.WenetInterface import WenetInterface
wenet_handler = WenetInterface(
    server_config.yaml_path,
    server_config.vocab_path,
    server_config.checkpoint,
    server_config.sample_rate,
    server_config.beam_size)

def decode_strict_wav(data, sr):
    pcm = data[44:]
    duration = len(pcm) / 2 / sr
    return pcm, duration

@server.route('/asr', methods=['POST', 'GET'])
def ASRRequest():
    if request.method == 'POST':
        request_id = 'request-' + str(uuid.uuid1())
        ret = {'ret_code': 1, 'audio_duration': 0, 'sn': ''}
        status_code = 1
        try:
            try:
                ip = request.headers['X-Forwarded-For']
            except Exception:
                ip = ""
            info_json = {'forward-ip': ip,
                         'remote_addr': request.remote_addr,
                         'url': request.url}
            info = json.dumps(info_json)
            print(info)
            sample_rate = int(request.values.get('sample_rate', None))
            if sample_rate is None:
                status_code = -300
                raise Exception('need param: sample_rate')
            audio_file = dict(request.files)
            audio_data = request.data
            if len(audio_file) <= 0 and len(audio_data) <= 0:
                status_code = -400
                raise Exception('No selected audio')
            else:
                if len(audio_file) > 0:
                    for _, audio in audio_file.items():
                        if isinstance(audio, list):
                            data = bytes(audio[0].read())
                        else:
                            data = bytes(audio.read())
                else:
                    data = bytes(audio_data)
                if len(data) > 0:
                    ret['sn'] = request_id
                    decode_start = time.time()
                    pcm, duration = decode_strict_wav(data, sample_rate)
                    status, result = wenet_handler.recognize(
                        pcm, sample_rate)
                    if not status:
                        status_code = -200
                        raise Exception(result)
                    ret['result'] = result
                    decode_elasped = time.time() - decode_start
                    if duration > 0:
                        decode_rtf = float(decode_elasped / duration)
                    else:
                        decode_rtf = 1000
                    ret['rtf'] = decode_rtf
                    ret['duration'] = float(duration)
                    ret['ret_code'] = status
                else:
                    status_code = -100
                    raise Exception(
                        'Not allowed audio file, upload amr or wav file please')
        except Exception as e:
            if status_code > -1000:
                ret['ret_code'] = status_code
            else:
                ret['ret_code'] = -2000
            ret['ret_msg'] = str(e)
        finally:
            ret_str = json.dumps(ret, ensure_ascii=False)
            print(ret_str)
            return ret_str
