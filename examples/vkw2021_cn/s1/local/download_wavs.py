#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2021 Tencent Inc. (Author: Yougen Yuan).                            
# Apach 2.0 

import argparse
import sys
import os
import random

def read_wav_scp(wav_scp_file):                                                 
    ###"00000001 wget -q -O - http://10.209.20.140/check_cn_312report/00001.wav |"
    wavscp_dict = {}                                                            
    for line in open(wav_scp_file, mode="r", encoding="utf8"):                  
        line = line.split("\n")[0].split()                                      
        wavscp_dict[line[0]] = line[1:]  ####" ".join(line[1:])                 
    return wavscp_dict

#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(description="", add_help=False)
    parser.add_argument("wavscp", type=str, help="the archive filename")
    parser.add_argument("datadir", type=str, help="the archive filename")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()
    print("Reading: ", args.wavscp)
    wavscp_dict = read_wav_scp(args.wavscp)
    wavscp_list = list(wavscp_dict.keys()) 
    print("wavscp_list has the size of %d" %(len(wavscp_list)))
    os.system("mkdir -p "+args.datadir) 
    for utt in wavscp_list:
        #print(wavscp_dict[utt])
        if len(wavscp_dict[utt]) == 1:
            #command="wget -q -O - \""+wavscp_dict[utt][0]+"\" | sox -t wav -r 16000 -c 1 -b 16 - -t wav "+os.path.join(args.datadir, utt)+".wav"
            command="wget -q -O - \""+wavscp_dict[utt][0]+"\" | ffmpeg -i - -ac 1 -ar 16000 "+os.path.join(args.datadir, utt)+".wav"
            #print(command)
            os.system(command)
            #break
        elif wavscp_dict[utt][0] == "wget" and wavscp_dict[utt][-1] == "|":
            command= " ".join(wavscp_dict[utt])+" sox -t wav -r 16k -c 1 - -t wav "+os.path.join(args.datadir, utt)+".wav"
            #print(command)
            os.system(command)
        elif wavscp_dict[utt][0] == "/usr/bin/sox" and wavscp_dict[utt][-1] == "|":
            command= " ".join(wavscp_dict[utt][:-5])+" "+os.path.join(args.datadir, utt)+".wav " + " ".join(wavscp_dict[utt][-4:-1])
            #print(command)
            os.system(command)
        elif wavscp_dict[utt][0] == "ffmpeg" and wavscp_dict[utt][-1] == "|":
            command= " ".join(wavscp_dict[utt][:-5])+" "+os.path.join(args.datadir, utt)+".wav " + " ".join(wavscp_dict[utt][-4:-1])
            #print(command)
            os.system(command)
        else:
            print("Not valid format")
            break
if __name__ == "__main__":
    main()
