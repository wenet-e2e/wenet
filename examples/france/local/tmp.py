import numpy as np
import os
import pandas as pd
import sys
args = sys.argv
src_dir = args[1]
tsv_file = src_dir+"/"+args[2]+".tsv"
output_dir = args[3]
import re,string

def process(src_str):
    punc = '~`!#$%^&*()_+-=|\';":/.,?><~·！@#￥%……&*（）——+-=“：’；、。，？》《{}'
    return(re.sub(r"[%s]+" %punc, "",src_str)).upper()
for file_path in os.listdir(src_dir+"/clips"):
    if(os.path.exists(src_dir+"/wavs/"+file_path.split('.')[0]+".wav")):
        continue
    os.system("ffmpeg -i {0} -ac 1 -ar 16000 -f wav {1}".format(src_dir+"/clips/"+file_path, src_dir+"/wavs/"+file_path.split('.')[0]+".wav"))

tsv_content = pd.read_csv(tsv_file, sep="\t")
print(tsv_content.head(5))
path_list = tsv_content["path"]
sentence = tsv_content["sentence"]
client_list = tsv_content["client_id"]
print(sentence[0])
scp_file = open(output_dir+"/wav.scp","w")
text_file = open(output_dir+"/text","w")
utt2spk = open(output_dir+"/utt2spk", "w")
for i in range(len(path_list)):
    wav_file = src_dir+"/wavs/"+path_list[i].split(".")[0]+".wav"
    now_sentence = process(sentence[i])
    scp_file.writelines(path_list[i].split(".")[0]+" "+wav_file+"\n")
    text_file.writelines(path_list[i].split(".")[0]+" "+now_sentence+"\n")
    utt2spk.writelines(path_list[i].split(".")[0]+" "+client_list[i]+"\n")
scp_file.close()
text_file.close()
utt2spk.close()

