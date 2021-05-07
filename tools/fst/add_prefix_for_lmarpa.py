#!/usr/bin/env python3
# encoding: utf-8

# Author: sxc19@mails.tsinghua.edu.cn (Xingchen Song)
import sys

# sys.argv[1]: prefix string, default: "▁"
# sys.argv[2]: input arpa file, default: ${lm}/lm.arpa.tmp
# sys.argv[3]: output arpa file, default: ${lm}/lm.arpa

prefix = sys.argv[1]
print("add prefix \"{}\" for each word, i.e., from \"hello\" to \"{}hello\"".format(
    prefix, prefix))

with open(sys.argv[2], 'r', encoding='utf8') as fin, \
        open(sys.argv[3], 'w', encoding='utf8') as fout:
    line = fin.readline()
    while (line):
        line = line.strip()
        prob = line.split('\t')[0]
        try:
            prob = float(prob)
            words = line.split('\t')[1]
            back_prob = line.split('\t')[2] if (
                len(line.split('\t')) > 2) else None
            words = " ".join([prefix + word if (word[0] != '<') else word
                              for word in words.split()])
            fout.write('{}\t{}'.format(prob, words))
            if back_prob is not None:
                fout.write('\t{}\n'.format(back_prob))
            else:
                fout.write('\n')
        except ValueError:
            fout.write('{}\n'.format(line))
        line = fin.readline()
