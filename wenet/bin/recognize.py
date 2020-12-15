# Copyright 2020 Mobvoi Inc. All Rights Reserved.
# Author: binbinzhang@mobvoi.com (Binbin Zhang)

from __future__ import print_function

import argparse
import copy
import logging
import os
import sys

import yaml
import torch
from torch.utils.data import DataLoader

from wenet.dataset.dataset import CollateFunc, AudioDataset
from wenet.transformer.encoder import TransformerEncoder
from wenet.transformer.encoder import ConformerEncoder
from wenet.transformer.decoder import TransformerDecoder
from wenet.transformer.ctc import CTC
from wenet.transformer.asr_model import ASRModel
from wenet.utils.checkpoint import load_checkpoint

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--test_data', required=True, help='test data file')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--cmvn', default=None, help='global cmvn file')
    parser.add_argument('--dict', required=True, help='dict file')
    parser.add_argument('--beam_size',
                        type=int,
                        default=10,
                        help='beam size for search')
    parser.add_argument('--penalty',
                        type=float,
                        default=0.0,
                        help='length penalty')
    parser.add_argument('--result_file', required=True, help='asr result file')
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='asr result file')
    parser.add_argument('--mode',
                        choices=[
                            'attention', 'ctc_greedy_search',
                            'ctc_prefix_beam_search', 'attention_rescoring'],
                        default='attention',
                        help='decoding mode')
    parser.add_argument('--ctc_weight',
                        type=float,
                        default=0.0,
                        help='ctc weight for attention rescoring decode mode')
    parser.add_argument('--decoding_chunk_size',
                        type=int,
                        default=-1,
                        help='''decoding chunk size,
                                <0: for decoding, use full chunk.
                                >0: for decoding, use fixed chunk size as set.
                                0: used for training, it's prohibited here''')
    parser.add_argument('--simulate_streaming',
                        action='store_true',
                        help='simulate streaming inference')
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    if args.mode in ['ctc_prefix_beam_search', 'attention_rescoring'
                     ] and args.batch_size > 1:
        logging.fatal(
            'decoding mode {} must be running with batch_size == 1'.format(
                args.mode))
        sys.exit(1)

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin)

    # Init dataset and data loader
    test_collate_conf = copy.copy(configs['collate_conf'])
    test_collate_conf['spec_aug'] = False
    test_collate_func = CollateFunc(**test_collate_conf, cmvn=args.cmvn)
    dataset_conf = configs.get('dataset_conf', {})
    dataset_conf['batch_size'] = args.batch_size
    dataset_conf['sort'] = False
    test_dataset = AudioDataset(args.test_data, **dataset_conf)
    test_data_loader = DataLoader(test_dataset,
                                  collate_fn=test_collate_func,
                                  shuffle=False,
                                  batch_size=1,
                                  num_workers=0)

    # Init transformer model
    input_dim = test_dataset.input_dim
    vocab_size = test_dataset.output_dim
    encoder_type = configs.get('encoder', 'conformer')
    if encoder_type == 'conformer':
        encoder = ConformerEncoder(input_dim, **configs['encoder_conf'])
    else:
        encoder = TransformerEncoder(input_dim, **configs['encoder_conf'])
    decoder = TransformerDecoder(vocab_size, encoder.output_size(),
                                 **configs['decoder_conf'])
    ctc = CTC(vocab_size, encoder.output_size())
    model = ASRModel(
        vocab_size=vocab_size,
        encoder=encoder,
        decoder=decoder,
        ctc=ctc,
        **configs['model_conf'],
    )

    # Load dict
    char_dict = {}
    with open(args.dict, 'r') as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            char_dict[int(arr[1])] = arr[0]
    eos = len(char_dict) - 1

    load_checkpoint(model, args.checkpoint)
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = model.to(device)

    model.eval()
    with torch.no_grad(), open(args.result_file, 'w') as fout:
        for batch_idx, batch in enumerate(test_data_loader):
            keys, feats, target, feats_lengths, target_lengths = batch
            feats = feats.to(device)
            target = target.to(device)
            feats_lengths = feats_lengths.to(device)
            target_lengths = target_lengths.to(device)
            if args.mode == 'attention':
                hyps = model.recognize(
                    feats,
                    feats_lengths,
                    beam_size=args.beam_size,
                    decoding_chunk_size=args.decoding_chunk_size,
                    simulate_streaming=args.simulate_streaming)
                hyps = [hyp.tolist() for hyp in hyps]
            elif args.mode == 'ctc_greedy_search':
                hyps = model.ctc_greedy_search(
                    feats,
                    feats_lengths,
                    decoding_chunk_size=args.decoding_chunk_size,
                    simulate_streaming=args.simulate_streaming)
            # ctc_prefix_beam_search and attention_rescoring only return one
            # result in List[int], change it to List[List[int]] for compatible
            # with other batch decoding mode
            elif args.mode == 'ctc_prefix_beam_search':
                assert (feats.size(0) == 1)
                hyp = model.ctc_prefix_beam_search(
                    feats,
                    feats_lengths,
                    args.beam_size,
                    decoding_chunk_size=args.decoding_chunk_size,
                    simulate_streaming=args.simulate_streaming)
                hyps = [hyp]
            elif args.mode == 'attention_rescoring':
                assert (feats.size(0) == 1)
                hyp = model.attention_rescoring(
                    feats,
                    feats_lengths,
                    args.beam_size,
                    decoding_chunk_size=args.decoding_chunk_size,
                    ctc_weight=args.ctc_weight,
                    simulate_streaming=args.simulate_streaming)
                hyps = [hyp]
            for i, key in enumerate(keys):
                content = ''
                for w in hyps[i]:
                    if w == eos:
                        break
                    content += char_dict[w]
                logging.info('{} {}'.format(key, content))
                fout.write('{} {}\n'.format(key, content))
