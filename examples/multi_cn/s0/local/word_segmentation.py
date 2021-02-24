#!/usr/bin/env python3
# encoding=utf-8

import sys
import jieba

for line in sys.stdin:
  blks = str.split(line)
  out_line = blks[0]
  for i in range(1, len(blks)):
    for j in jieba.cut(blks[i], cut_all=False):
      out_line += " " + j
  print(out_line)