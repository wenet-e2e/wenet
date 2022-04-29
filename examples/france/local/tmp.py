#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import re
def process(src_str):
    punc = '~`!#$%^&*()_+-=|\';":/.,?><~·！@#￥%……&*（）——+-=“：’；、。，？》《{}'
    return re.sub(r"[{0}]+".format(punc), "", src_str).upper()

if __name__ == '__main__':
    src_dir = sys.argv[1]
    tsv_file = src_dir + "/" + sys.argv[2] + ".tsv"
    output_dir = sys.argv[3]
    for file_path in os.listdir(src_dir + "/clips"):
        if(os.path.exists(src_dir + "/wavs/" + file_path.split('.')[0] + ".wav")):
            continue
        t_str = src_dir + "/clips/" + file_path
        tt_str = src_dir + "/wavs/" + file_path.split('.')[0] + ".wav"
        os.system("ffmpeg -i {0} -ac 1 -ar 16000 -f wav {1}".format(t_str, tt_str))
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
        now_sentence = process(sentence[i])
        wav_file = src_dir + "/wavs/" + temple_str + ".wav"
        scp_file.writelines(temple_str + " " + wav_file + "\n")
        text_file.writelines(temple_str + " " + now_sentence + "\n")
        utt2spk.writelines(temple_str + " " + client_list[i] + "\n")
    scp_file.close()
    text_file.close()
    utt2spk.close()
