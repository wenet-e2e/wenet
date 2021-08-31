import torch
import numpy as np
import struct
import logging
import http
import urllib3
import torch.distributed as dist
import random

from torch.utils.data import IterableDataset
from torch.nn.utils.rnn import pad_sequence
from wenet.dataset.dataset import _spec_augmentation, _spec_substitute 

logging.basicConfig(level=logging.DEBUG)


class MxfeatsIterableDataset(IterableDataset):
    def __init__(self, client, mxfeats_lst, feats_mean, feats_var, 
                 feat_dim=80, buffer_size=10000):
        super(MxfeatsIterableDataset).__init__()
        self.feat_dim = feat_dim
        self.mxfeats_lst = mxfeats_lst
        self.feats_mean = feats_mean
        self.feats_var = feats_var
        self.client = client
        self._buffer = []
        self.buffer_size = buffer_size
        
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single process
            num_workers = 1
            worker_rank = 0
        else:
            # In a worker process
            num_workers = worker_info.num_workers
            worker_rank = worker_info.id

        if not dist.is_initialized():
            gpu_rank = 0 
            gpu_size = 1 
        else:
            gpu_rank = dist.get_rank()
            gpu_size = dist.get_world_size()
            logging.info("rank infor: %d %d\n" % (gpu_rank, gpu_size))

        skip = max(1, gpu_size * num_workers)
        offset = gpu_rank * num_workers + worker_rank
        logging.info("worker_rank is %d\n" % worker_rank)
        if offset >= len(self.mxfeats_lst):
            logging.info("List length is less than offset!!!")
            offset = 0

        data_lst = self.mxfeats_lst[offset::skip]
       
        client = self.client 
        feat_dim = self.feat_dim
        for i in range(len(data_lst)):
            try:
                mxfeats = client.read(data_lst[i])
            except http.client.IncompleteRead:
                logging.info("Caught http.client.IncompleteRead when reading %s" % data_lst[i])
                continue
            except urllib3.exceptions.NewConnectionError:
                logging.info("Caught urllib3.exceptions.NewConnectionError when reading %s" % data_lst[i])
                continue
            except Exception as e:
                logging.info("Caught unknown exception when reading %s" % data_lst[i])
                logging.info(e)
                continue
            tot_byte_num = len(mxfeats)
            idx = 0
            samples = 0
            # Parse data format
            # Data format: 4 bytes magic num + 4 bytes for data bytes number + 
            #              frm_num of int32_t labels + 
            #              frm_num of float32*feat_dim features +
            #              8 btyes placeholder
            while idx < tot_byte_num:
                byte_num = struct.unpack('<I', mxfeats[idx + 4:idx + 8])[0]
                frm_num = int((byte_num - 8) / (feat_dim + 1) / 4)
                fmt = '<' + str(frm_num) + 'i' + str(frm_num * feat_dim) + 'f'
                ali_feats = struct.unpack(fmt, mxfeats[idx + 8:idx + byte_num])
                ali_data = np.array(ali_feats[:frm_num])
                ali_num = np.count_nonzero(ali_data)
                feats_data = np.reshape(np.array(ali_feats[frm_num:]), (frm_num, feat_dim))
                feats_data = (feats_data + self.feats_mean) * self.feats_var
                # Ignore too long sentences
                if frm_num > 500 or ali_num > 100:
                    idx += (8 + byte_num)
                    continue
                if len(self._buffer) == self.buffer_size:
                    rand_idx = random.randint(0, self.buffer_size - 1)
                    yield self._buffer[rand_idx]
                    self._buffer[rand_idx] = (feats_data, ali_data[:ali_num]) 
                else:
                    self._buffer.append((feats_data, ali_data[:ali_num]))
                idx += (8 + byte_num)  # 4 bytes magic and 4 bytes byte_num
                samples += 1
        while self._buffer:
            yield self._buffer.pop() 


class CollateFunc(object):
    
    def __init__(self,
                 feature_dither=0.0,
                 spec_aug=False,
                 spec_aug_conf=None,
                 spec_sub=False,
                 spec_sub_conf=None,
                 raw_wav=False):
        
        self.feature_dither = feature_dither
        self.spec_aug = spec_aug
        self.spec_aug_conf = spec_aug_conf
        self.spec_sub = spec_sub
        self.spec_sub_conf = spec_sub_conf

    def __call__(self, batch):

        # optional feature dither d ~ (-a, a) on fbank feature
        # a ~ (0, 0.5)
        if self.feature_dither != 0.0:
            a = random.uniform(0, self.feature_dither)
            xs = [x[0] + (np.random.random_sample(x[0].shape) - 0.5) * a for x in batch]
        else:
            xs = [x[0] for x in batch]

        # optinoal spec substitute
        if self.spec_sub:
            xs = [_spec_substitute(x, **self.spec_sub_conf) for x in xs]

        # optinoal spec augmentation
        if self.spec_aug:
            xs = [_spec_augmentation(x, **self.spec_aug_conf) for x in xs]

        xs_lengths = torch.from_numpy(np.array([x.shape[0] for x in xs], 
                                      dtype=np.int32))
        xs_pad = pad_sequence([torch.from_numpy(x).float() for x in xs], True, 0)
        
        ys_lengths = torch.from_numpy(np.array([y[1].shape[0] for y in batch],
                                      dtype=np.int32))
        ys_pad = pad_sequence([torch.from_numpy(y[1]).int() for y in batch], True, -1)

        return xs_pad, ys_pad, xs_lengths, ys_lengths
