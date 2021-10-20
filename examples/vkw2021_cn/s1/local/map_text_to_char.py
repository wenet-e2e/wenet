#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2021 Tencent Inc. (Author: Yougen Yuan).
# Apach 2.0

import argparse
import sys


def map_words2char(wordlist):
    worddict = {}
    for line in open(wordlist, mode="r", encoding="utf8"):
        line = line.split("\n")[0].split()
        if line[0] not in worddict:
            worddict[line[0]] = [i for i in line[0]] 
    return worddict


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0], add_help=False)
    parser.add_argument("text", type=str, help="the archive filename")
    parser.add_argument("dicts", type=str, help="the archive filename")
    parser.add_argument("text_new", type=str, help="the archive filename")
    if len(sys.argv) != 4:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#


def main():
    args = check_argv()
    print("Reading:", args.dicts)
    dicts = []
    for line in open(args.dicts, mode="r", encoding="utf8"):
        line = line.split("\n")[0].split(" ")
        dicts.append(line[0])

    with open(args.text_new, mode="w", encoding="utf8") as f:
        for line in open(args.text, mode="r", encoding="utf8"):
            line = line.split("\n")[0].split()
            temp = []
            temp2 = []
            flag = True
            for i in "".join(line[1:]):
                if i in dicts:
                    if flag:
                        temp.append(i)
                        #print(i, flag, temp)
                    else:
                        temp.append("".join(temp2))
                        temp.append(i)
                        flag = True
                        temp2 = []
                        #print(i, flag, temp)
                else:
                    if flag:
                        temp2.append(i)
                        flag = False
                        #print(i, flag, temp)
                    else:
                        temp2.append(i)
                        #print(i, flag, temp)
            #print(flag, temp2)
            if len(temp2) > 0:
                if flag:
                    pass
                else:
                    temp.append("".join(temp2))
            else:
                pass
            f.write(line[0]+" " + " ".join(temp)+"\n")
if __name__ == "__main__":
    main()
