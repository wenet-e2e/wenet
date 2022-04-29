#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os


def sph2pipe_wav(in_wav, tmp_out_wav, out_wav):
    with open(in_wav, 'r', encoding='utf-8') as in_f:
        with open(tmp_out_wav, 'w', encoding='utf-8') as tmp_out_f:
            with open(out_wav, 'w', encoding='utf-8') as out_f:
                for line in in_f:
                    _tmp = line.strip().split(' ')
                    wav_out_path = _tmp[4]
                    wav_out_path = wav_out_path.split('/')
                    wav_out_path[-4] = wav_out_path[-4] + '_pipe'
                    if not os.path.exists('/'.join(wav_out_path[:-1])):
                        os.makedirs('/'.join(wav_out_path[:-1]))
                    wav_out_path = '/'.join(wav_out_path)
                    tmp_out_f.write(' '.join(_tmp[1:5]) + ' ' + wav_out_path +
                                    '\n')
                    out_f.write(_tmp[0] + ' ' + wav_out_path + '\n')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('wrong input parameter')
        raise NotImplementedError(len(sys.argv))
    src_dir = sys.argv[1]
    tsv_file = src_dir + "/" + sys.argv[2] + ".tsv"
    output_dir = sys.argv[3]
    import pandas
    tsv_content = pandas.read_csv(tsv_file, sep="\t")
    path_list = tsv_content["path"]
    sentence = tsv_content["sentence"]
    client_list = tsv_content["client_id"]
    scp_file = open(output_dir + "/wav.scp", "w")
    text_file = open(output_dir + "/text", "w")
    utt2spk = open(output_dir + "/utt2spk", "w")
    for i in range(len(path_list)):
        wav_file = src_dir + "/wavs/" + \
                path_list[i].split(".")[0] + ".wav"
        now_sentence = sentence[i]
        scp_file.writelines(path_list[i].split(".")[0] + " " + wav_file + "\n")
        text_file.writelines(path_list[i].split(".")[0] + " " + now_sentence + "\n")
        utt2spk.writelines(path_list[i].split(".")[0] + " " + client_list[i] + "\n")
    scp_file.close()
    text_file.close()
    utt2spk.close()
