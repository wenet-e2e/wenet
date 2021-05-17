#coding=utf-8

class ServerConfig:
    def __init__(self) -> None:
        self.port = 10073
        self.crt = '/storage08/gzdinghanyu/code/pytorch/ASR/Gitrepo/wenet/flask/ssl/server.crt'
        self.key = '/storage08/gzdinghanyu/code/pytorch/ASR/Gitrepo/wenet/flask/ssl/server.key'
        self.sample_rate = 16000
        self.checkpoint = '/storage08/gzdinghanyu/code/pytorch/ASR/Gitrepo/wenet/exp/20210204_conformer_exp/final.pt'
        self.yaml_path = '/storage08/gzdinghanyu/code/pytorch/ASR/Gitrepo/wenet/exp/20210204_conformer_exp/train.yaml'
        self.vocab_path = '/storage08/gzdinghanyu/code/pytorch/ASR/Gitrepo/wenet/exp/20210204_conformer_exp/words.txt'
        self.beam_size = 10