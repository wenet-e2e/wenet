# coding=utf-8

from wenet.transformer.asr_model import init_asr_model
from wenet.utils.checkpoint import load_checkpoint
import numpy as np
import torch
import torchaudio
import os
import re
import yaml
import librosa

# deploy on cpu
# decode mode is attention rescoring
class WenetInterface:
    def __init__(self,
                 yaml_path,
                 vocab_path,
                 checkpoint,
                 sample_rate=8000,
                 beam_size=10):
        os.environ['PYTORCH_JIT'] = '0'
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['LRU_CACHE_CAPACITY'] = '1'
        torch.set_num_threads(1)
        torch.backends.cudnn.enabled = False
        os.environ['CUDA_VISIBLE_DEVICES'] = ""

        # load config
        with open(yaml_path, 'r') as fin:
            self.configs = yaml.load(fin)

        # load vocab
        self.char_dict = {}
        with open(vocab_path, 'r') as fin:
            for line in fin:
                arr = line.strip().split()
                assert len(arr) == 2
                self.char_dict[int(arr[1])] = arr[0]
        self.eos = len(self.char_dict) - 1

        self.model = init_asr_model(self.configs)
        load_checkpoint(self.model, checkpoint)
        self.model.to(torch.device('cpu'))
        self.model.eval()

        self.sample_rate = sample_rate

        # control max memory usage
        self.beam_size = beam_size
        self.max_decode_length = 5000
        self.trunc_length = 4000

        self.feat_hparams = self.configs['collate_conf']['feature_extraction_conf']

        print(self.feat_hparams)
        print("finish init wenet interface")

    def recognize(self, pcm, sr):
        try:
            with torch.no_grad():
                all_hyps = []
                status, feat = self.compute_feature(pcm, sr)
                if not status:
                    raise Exception(feat)
                batchs = self.split_feat(feat)
                for item in batchs:
                    feats, feats_lengths = item
                    feats_lengths = feats_lengths.view(-1)
                    hyp = self.model.attention_rescoring(
                        feats, feats_lengths, self.beam_size,
                        decoding_chunk_size=-1,
                        num_decoding_left_chunks=-1,
                        ctc_weight=0.5,
                        simulate_streaming=False)
                    all_hyps.extend(list(hyp))
                content = ''
                for w in all_hyps:
                    if w == self.eos:
                        break
                    content += self.char_dict[w]
                content = content.replace('▁', ' ')
                return True, content
        except Exception as e:
            return False, str(e)

    def change_sample_rate(self, src_data, src_sample_rate, dst_sample_rate):
        '''
            input type: numpy.ndarray or bytes
            out type: bytes
        '''
        if isinstance(src_data, bytes):
            src_data = np.frombuffer(
                src_data, dtype=np.int16).astype(np.float32)
        dst_data = librosa.resample(
            src_data, src_sample_rate, dst_sample_rate, res_type='polyphase')
        dst_data = dst_data.astype(np.int16).tobytes()
        return dst_data

    # modify this function by self
    def compute_feature(self, pcm, sr):
        if sr != self.sample_rate:
            pcm = self.change_sample_rate(pcm, sr, self.sample_rate)
        if isinstance(pcm, bytes):
            pcm = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        if isinstance(pcm, np.ndarray):
            audios = torch.FloatTensor(pcm)
        else:
            audios = pcm
        if len(audios.size()) > 1:
            pass
        else:
            audios = audios.unsqueeze(0)
        fbank = self.fbank(audios)
        return True, fbank

    def split_feat(self, feat):
        if torch.is_tensor(feat):
            feat = feat.detach().cpu().numpy()
        feat_length = feat.shape[0]
        start = 0
        batchs = []
        while True:
            if feat_length - start < self.max_decode_length:
                sub_feat = np.asarray([feat[start:]])
                sub_feat_lens = np.asarray(
                    [[sub_feat.shape[1]]], dtype=np.int32)
                sub_feats_torch = torch.from_numpy(sub_feat).float()
                sub_feat_lens_torch = torch.from_numpy(
                    sub_feat_lens).long()
                batchs.append([sub_feats_torch, sub_feat_lens_torch])
                break
            else:
                sub_feat = np.asarray(
                    [feat[start:(start + self.trunc_length)]])
                start += self.trunc_length
                sub_feat_lens = np.asarray(
                    [[sub_feat.shape[1]]], dtype=np.int32)
                sub_feats_torch = torch.from_numpy(sub_feat).float()
                sub_feat_lens_torch = torch.from_numpy(
                    sub_feat_lens).long()
                batchs.append([sub_feats_torch, sub_feat_lens_torch])
        return batchs

    def fbank(self, pcm):
        feat = torchaudio.compliance.kaldi.fbank(
            pcm,
            num_mel_bins=self.feat_hparams['mel_bins'],
            frame_length=self.feat_hparams['frame_length'],
            frame_shift=self.feat_hparams['frame_shift'],
            dither=0.0,
            energy_floor=0.0,
            sample_frequency=self.sample_rate)
        return feat

    def normalize(self, feature):
        if isinstance(feature, np.ndarray):
            feature = torch.from_numpy(feature).float()
        mean = torch.mean(feature)
        std = torch.std(feature)
        if std < 0.5:
            return False, "the variance is too low, maybe sil or noise"
        else:
            return True, ((feature - mean) / std)

    def read_label_dict(self, path):
        ret_dict = {}
        for line in open(path, 'rt'):
            items = re.split(r'\s+', line.strip())
            phone = items[0]
            _id = int(items[1])
            ret_dict[phone] = _id
        return ret_dict
