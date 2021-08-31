#!/usr/bin/python
# -*- coding: utf-8 -*-
import hashlib
import os
import time
import struct
import numpy as np

from hdfs import InsecureClient


class HdfsCli:

    HDFS_NODES = ['cpu001.cc', 'cpu002.cc']

    def __init__(self):
        node = self.get_ha_node(self.HDFS_NODES)
        user = 'xxx'
        self.client = InsecureClient('http://' + node + ':50010', user=user)
        self.path = '/user/xxx'
        self.prefix = 'hdfs://' + node + ':8010'

    def get_ha_node(self, hdfs_nodes):
        for node in hdfs_nodes:
            hadoop = InsecureClient('http://' + node + ':50010')
            try:
                hadoop.status('/')
                return node
            except Exception:
                continue
        return ''
 
    def upload(self, local_folder, data=None):
        if data is None:
            data = str(int(time.time()))
 
            remote_folder = self.path + '/' + self.__getname(data)
            if self.client.status(remote_folder, strict=False) is None:
                self.client.makedirs(remote_folder, permission=755)
            try:
                ret = self.client.upload(remote_folder, local_folder,
                                         n_threads=5)
                return ret
            except Exception:
                return None

    def upload_file(self, remote_folder, local_file):
 
        if self.client.status(remote_folder, strict=False) is None:
            self.client.makedirs(remote_folder, permission=755)
        #try:
        #    ret = self.client.upload(remote_folder, local_folder, n_threads=5)
        #    return ret
        #except Exception:
        #    print("upload file failed!!!")
        #    return None
        ret = self.client.upload(remote_folder, local_file, n_threads=5, overwrite=True)
        return ret
 
    def download(self, remote_folder, local_folder):
        if os.path.exists(local_folder) is None:
            os.makedirs(local_folder)
 
            if self.client.status(remote_folder, strict=False) is None:
                return None
 
            try:
                ret = self.client.download(remote_folder, local_folder,
                                           n_threads=5)
                return ret
            except Exception:
                return None

    def read(self, path):
        with self.client.read(path) as reader:
            content = reader.read()
            return content
 
    def list(self, remote_folder):
        ret = self.client.list(remote_folder)
        return ret
 
    def host(self):
        return self.prefix
 
    def __getname(self, data):
        m = hashlib.md5()
        m.update(data)
        return m.hexdigest()


if __name__ == '__main__':
    t1 = time.time() 
    feat_dim = 80
    client = HdfsCli()
    mxfeats = client.read('xfeats_92')
    tot_byte_num = len(mxfeats)
    idx = 0
    t2 = time.time()
    samples = 0
    times = 0
    while idx < tot_byte_num:
        t2_1 = time.time()
        byte_num = struct.unpack('<I', mxfeats[idx + 4:idx + 8])[0]
        t2_2 = time.time()
        frm_num = int((byte_num - 8) / (feat_dim + 1) / 4)
        fmt = '<' + str(frm_num) + 'i' + str(frm_num * feat_dim) + 'f'
        ali_feats = struct.unpack(fmt, mxfeats[idx + 8:idx + byte_num])
        t2_3 = time.time()
        ali_data = np.array(ali_feats[:frm_num])
        t2_4 = time.time()
        feats_data = np.array(ali_feats[frm_num:])
        t2_5 = time.time()
        feats_data = np.reshape(feats_data, (frm_num, feat_dim))
        t2_6 = time.time()
        idx += (8 + byte_num)  # 4 bytes magic and 4 bytes byte_num
        samples += 1
        #print(ali_data)
        #print(feats_data)
        times += 1
        print(times, t2_2 - t2_1, t2_3 - t2_2, t2_4 - t2_3, t2_5 - t2_4, t2_6 - t2_5)
    t3 = time.time()
