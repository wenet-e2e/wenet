# coding=utf-8
__author__ = 'day9011'


import os
import torch
os.environ['PYTORCH_JIT'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['LRU_CACHE_CAPACITY'] = '1'
torch.set_num_threads(1)
torch.backends.cudnn.enabled = False

from config.config import ServerConfig
server_config = ServerConfig()

from api import server
server.debug = False

if __name__ == '__main__':
    port = int(server_config.port)
    host = '0.0.0.0'
    server.run(host=host, port=port)
