# coding: utf-8
# Author：WangTianRui
# Date ：2021/5/15 21:17
import numpy as np
import torch, os, json
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as torchF
import feature_and_aug as feature_and_aug
from torch.utils.data import DataLoader


class AudioDataset(Dataset):
    def __init__(self, json_file, data_home, only_asr,
                 max_length, min_length,
                 token_max_length, token_min_length, sort,
                 speed_perturb, spec_aug, spec_aug_conf, wav_dither, stft_conf, fbank_conf):
        super(AudioDataset, self).__init__()
        self.data_home = data_home
        self.only_asr = only_asr
        self.json_file = json_file  # format.data
        self.max_length = max_length
        self.min_length = min_length
        self.token_max_length = token_max_length
        self.token_min_length = token_min_length
        self.sort = sort
        datas = []
        with open(json_file, "r") as f:
            self.all_infos = json.load(f)
        for key in self.all_infos.keys():
            item = self.all_infos[key]
            asr_infos = item["asr_infos"]

            arr = asr_infos.strip().split('\t')
            if len(arr) != 7:
                continue
            wav_name = arr[0].split(":")[1]
            token_id = arr[5].split(':')[1]
            out_dim = int(arr[6].split(':')[1].split(',')[1])

            duration = int(float(arr[2].split(':')[1]) * 1000 / 10)  # 转为10ms为单位

            if duration > max_length or duration < min_length:
                continue
            token_len = len(token_id.split())
            if token_len > token_max_length or token_len < token_min_length:
                continue

            wav_path = os.path.join(self.data_home, self.all_infos[key]["noisy_path"])
            datas.append((wav_name, wav_path, duration, token_id))
            self.out_dim = out_dim

        if sort:
            datas = sorted(datas, key=lambda x: float(x[2]))  # 按duration排序
        self.all_datas = []
        self.max_audio_len = 0
        self.max_text_len = 0
        for item in datas:
            if float(item[2]) > self.max_audio_len:
                self.max_audio_len = float(item[2])
            if len(item[3]) > self.max_text_len:
                self.max_text_len = len(item[3])
            self.all_datas.append((item[0], item[1], item[3]))

        self.speed_perturb = speed_perturb
        self.spec_aug = spec_aug
        self.spec_aug_conf = spec_aug_conf
        self.wav_dither = wav_dither
        if self.spec_aug:
            self.spec_auger = feature_and_aug.SpecAug(**self.spec_aug_conf)

        # 针对AISHELL1最大长度不超过16s。其他数据集可以做成在计算CMVN时计算获取
        self.max_time_len = 16000 * 16 - stft_conf["win_hop"]
        # 窗长400，步长160，情况下最大的nframe数（可以大于，因为在后面读进模型后还会选取一次，参考126行的测试系统类）
        self.mask_max_len = 1600
        self.win_len = stft_conf["win_len"]
        self.hop = stft_conf["win_hop"]
        self.f_bins = fbank_conf["n_bin"]

    def __len__(self):
        return len(self.all_datas)

    def __getitem__(self, idx):
        """
        读取pad后的wave信号以及label
        根据需求计算spec_aug的mask，将mask作为输入送出去
        """
        mask = None
        wav_info = self.all_datas[idx]
        speed = 1.0

        noisy_wav, sr = feature_and_aug.load_wav(wav_info[1], speed=speed, dither=self.wav_dither)
        # 计算frame的数量
        n_frame = int((len(noisy_wav) + 2 * (self.win_len - self.hop) - (self.win_len - 1) - 1) / self.hop + 1)
        fbank_len = torch.tensor(n_frame).to(torch.int32)
        if self.spec_aug:
            mask = self.spec_auger(t=n_frame, f=self.f_bins)
            mask = torchF.pad(mask, [0, 0, 0, int(self.mask_max_len - n_frame)])
        noisy_wav = torchF.pad(noisy_wav, [0, self.max_time_len - len(noisy_wav)])
        wav_name = wav_info[0]
        label = np.array(wav_info[2].split(), dtype=np.int32)
        label_len = torch.tensor(label.shape[0]).to(torch.int32)
        label = torch.tensor(
            np.pad(label, [0, int(self.max_text_len - len(label))], "constant", constant_values=-1),
            dtype=torch.int32)
        if self.spec_aug:
            return wav_name, (noisy_wav, mask), label, fbank_len, label_len
        else:
            return wav_name, noisy_wav, label, fbank_len, label_len


def init_dataloader(dataset_conf, data_home, batch_size, train_data_file, valid_data_file, num_worker,
                    speed_perturb, spec_aug, spec_aug_conf, wav_dither, stft_conf, fbank_conf, only_asr):
    assert os.path.exists(train_data_file) and os.path.exists(valid_data_file), "info path error"
    train_dataset = AudioDataset(json_file=train_data_file, data_home=data_home, **dataset_conf, only_asr=only_asr,
                                 speed_perturb=speed_perturb, spec_aug=spec_aug, spec_aug_conf=spec_aug_conf,
                                 wav_dither=wav_dither, stft_conf=stft_conf, fbank_conf=fbank_conf)
    valid_dataset = AudioDataset(json_file=valid_data_file, data_home=data_home, **dataset_conf, only_asr=only_asr,
                                 speed_perturb=speed_perturb, spec_aug=spec_aug, spec_aug_conf=spec_aug_conf,
                                 wav_dither=wav_dither, stft_conf=stft_conf, fbank_conf=fbank_conf)
    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size, num_workers=num_worker)
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=batch_size, num_workers=num_worker)
    return train_dataloader, valid_dataloader, train_dataset.out_dim


class System(nn.Module):
    def __init__(self, asr, stft_conf, fbank_conf):
        """
        一个示例模型
        conf 示例 yaml:
        stft_conf:
          win_len: 400
          win_hop: 160
          n_fft: 512
          win_type: povey
        fbank_conf:
          n_bin: 80
          sr: 16000.0
          low_freq: 20.0
          high_freq: 0.0
          vtln_low: 100.0
          vtln_high: -500.0
          vtln_wrap: 1.0
          use_log_fbank: true
          preemphasis_coefficient: 0.97
        """
        super(System, self).__init__()
        self.ASR = asr
        self.fbank_extractor = feature_and_aug.Features(**stft_conf, fbank_conf=fbank_conf)

    def forward(self, feats, fbank_t_len, text, text_len):
        if len(feats) == 2:
            wave, spec_mask = feats
            fbank = self.fbank_extractor(wave)
            fbank *= spec_mask
        else:
            wave = feats
            fbank = self.fbank_extractor(wave)

        order = torch.argsort(fbank_t_len, descending=False)
        max_fbank_len = torch.max(fbank_t_len)
        max_text_len = torch.max(text_len)
        fbank = fbank[order, :max_fbank_len].cuda()
        text = text[order, :max_text_len].cuda()
        fbank_t_len = fbank_t_len[order, ...].cuda()
        text_len = text_len[order, ...].cuda()
        return self.ASR(fbank, fbank_t_len, text, text_len)
