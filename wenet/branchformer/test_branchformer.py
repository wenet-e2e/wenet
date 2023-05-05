# Copyright (c) 2023 Voicecomm Inc (Kai Li)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import argparse
import yaml
from wenet.branchformer.encoder import BranchformerEncoder

def get_args():
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--config', required=True, help='config file')
    args = parser.parse_args()
    return args

def test_branchformer(configs):
    net = BranchformerEncoder(input_size=80, **configs['encoder_conf'])
    net.eval()
    feat = torch.randn(12,120,80)
    rtensor = torch.range(0, 22, 2)
    feat_len = torch.ones(12) * 120 - rtensor
    out = net(feat, feat_len)
    print(out[0])


if __name__ == '__main__':
    args = get_args()
    torch.manual_seed(557)
    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    test_branchformer(configs)