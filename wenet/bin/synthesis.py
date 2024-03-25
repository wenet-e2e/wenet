# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Xiaoyu Chen, Di Wu)
#               2023 WeNet Community (authors: Binbin Zhang)
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

import argparse
import json
import logging
import os
import yaml
import shutil

import torch
import torchaudio

from wenet.utils.init_model import init_model
from wenet.utils.init_tokenizer import init_tokenizer


def get_args():
    parser = argparse.ArgumentParser(description='recognize with your model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--result_dir', required=True, help='asr result file')
    parser.add_argument('--test_data', required=True, help='test data file')
    args = parser.parse_args()
    print(args)
    return args


def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    model, configs = init_model(args, configs)
    tokenizer = init_tokenizer(configs)

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = model.to(device)
    model.eval()

    os.makedirs(args.result_dir, exist_ok=True)

    with open(args.test_data, 'r') as fin:
        for i, line in enumerate(fin):
            obj = json.loads(line.strip())
            assert 'key' in obj
            assert 'wav' in obj
            assert 'txt' in obj
            key = obj['key']
            wav_file = obj['wav']
            txt = obj['txt']
            # stxt = obj['syn']
            txt = txt + ' ' + 'jin1 tian1 tian1 qi4 zen3 me yang4'
            # txt = txt + ' ' + txt
            print(key, wav_file, txt)
            wav, sample_rate = torchaudio.load(wav_file)
            ref_text = torch.tensor(tokenizer.tokenize(txt)[1],
                                    dtype=torch.long,
                                    device=device)
            batch = {}
            batch['pcm'] = wav[0, :].unsqueeze(0)
            batch['pcm_length'] = torch.tensor([wav.size(1)], dtype=torch.long)
            batch['target'] = ref_text.unsqueeze(0)
            batch['target_lengths'] = torch.tensor([ref_text.size(0)],
                                                   dtype=torch.long)
            with torch.no_grad():
                gen_wav, sample_rate = model.infer(batch, device)
            shutil.copy(wav_file, args.result_dir)
            save_path = os.path.join(args.result_dir,
                                     '{}.vqtts.wav'.format(key))
            torchaudio.save(save_path,
                            gen_wav.squeeze(0),
                            sample_rate,
                            encoding='PCM_S',
                            bits_per_sample=16)
            print('Save to ' + save_path)
            # break


if __name__ == '__main__':
    main()
