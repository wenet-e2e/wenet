import torch
import sys
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