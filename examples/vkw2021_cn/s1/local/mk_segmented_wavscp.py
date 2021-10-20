#!/usr/bin/env python
# -*- coding: utf-8 -*-                                                         
# Copyright 2021 Tencent Inc. (Author: Yougen Yuan).                            
# Apach 2.0 
import argparse
import sys

def read_segments(segments_file):                                               
    ###"00000001_0060_000284-000340_020587 00000001_0060 2.850 3.406"           
    segments_dict = {}                                                          
    for line in open(segments_file, mode="r", encoding="utf8"):                 
        line = line.split("\n")[0].split()                                      
        segments_dict[line[0]] = [line[1], float(line[2]), float(line[3])]      
    return segments_dict                                                        
                                                                                
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
    parser.add_argument("segments", type=str, help="the archive filename")
    parser.add_argument("wav_scp", type=str, help="the archive filename")
    parser.add_argument("segmented_wav_scp", type=str, help="the archive filename")
    if len(sys.argv) != 4:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    print("Reading:", args.segments)
    segments_dict = read_segments(args.segments)
    segments_list = list(segments_dict.keys())
    print("segments has the size of: ", len(segments_list))

    wav_scp_dict = read_wav_scp(args.wav_scp)
    wav_scp_list = list(wav_scp_dict.keys())
    print("wav scp has the size of: ", len(wav_scp_list))
   
    f = open(args.segmented_wav_scp, mode="w", encoding="utf8")
    for i in segments_list:
        if segments_dict[i][0] not in wav_scp_list:
            print(segments_dict[i], " is not found in wav.scp")
            continue
        duration = segments_dict[i][2] - segments_dict[i][1]
        url = " ".join(wav_scp_dict[segments_dict[i][0]])
        if url.startswith("wget"):
            f.write(i+" "+url+" sox -t wav - -r 16k -c 1 -t wav - trim {:.3f} {:.3f} |".\
format(segments_dict[i][1], duration)+"\n")
        elif url.startswith("ffmpeg"):
            f.write(i+" "+url+" sox -t wav - -r 16k -c 1 -t wav - trim {:.3f} {:.3f} |".\
format(segments_dict[i][1], duration)+"\n")
        else:
            f.write(i+" /usr/bin/sox -t wav "+ url +" -t wav - trim {:.3f} {:.3f} |".\
format(segments_dict[i][1], duration)+"\n")
    f.close()

if __name__ == "__main__":
    main()
