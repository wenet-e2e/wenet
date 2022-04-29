#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os


def sph2pipe_wav(in_wav, tmp_out_wav, out_wav):
    pass

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
        temple_str = path_list[i].split(".")[0]
        now_sentence = sentence[i]
        wav_file = src_dir + "/wavs/" + temple_str + ".wav"
        scp_file.writelines(temple_str + " " + wav_file + "\n")
        text_file.writelines(temple_str + " " + now_sentence + "\n")
        utt2spk.writelines(temple_str + " " + client_list[i] + "\n")
    scp_file.close()
    text_file.close()
    utt2spk.close()
