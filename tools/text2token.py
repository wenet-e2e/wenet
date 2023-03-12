#!/usr/bin/env python3

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
# Copyright 2021 JD AI Lab. All Rights Reserved. (authors: Lu Fan)
# Copyright 2021 Mobvoi Inc. All Rights Reserved. (Di Wu)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import print_function
from __future__ import unicode_literals

import argparse
import codecs
import re
import sys

is_python2 = sys.version_info[0] == 2


def exist_or_not(i, match_pos):
    start_pos = None
    end_pos = None
    for pos in match_pos:
        if pos[0] <= i < pos[1]:
            start_pos = pos[0]
            end_pos = pos[1]
            break

    return start_pos, end_pos

def seg_char(sent):
    pattern = re.compile(r'([\u4e00-\u9fa5])')
    chars = pattern.split(sent)
    chars = [w for w in chars if len(w.strip()) > 0]
    return chars

def get_parser():
    parser = argparse.ArgumentParser(
        description='convert raw text to tokenized text',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--nchar',
                        '-n',
                        default=1,
                        type=int,
                        help='number of characters to split, i.e., \
                        aabb -> a a b b with -n 1 and aa bb with -n 2')
    parser.add_argument('--skip-ncols',
                        '-s',
                        default=0,
                        type=int,
                        help='skip first n columns')
    parser.add_argument('--space',
                        default='<space>',
                        type=str,
                        help='space symbol')
    parser.add_argument('--bpe-model',
                        '-m',
                        default=None,
                        type=str,
                        help='bpe model for english part')
    parser.add_argument('--non-lang-syms',
                        '-l',
                        default=None,
                        type=str,
                        help='list of non-linguistic symobles,'
                        ' e.g., <NOISE> etc.')
    parser.add_argument('text',
                        type=str,
                        default=False,
                        nargs='?',
                        help='input text')
    parser.add_argument('--trans_type',
                        '-t',
                        type=str,
                        default="char",
                        choices=["char", "phn", "cn_char_en_bpe"],
                        help="""Transcript type. char/phn. e.g., for TIMIT
                             FADG0_SI1279 -
                             If trans_type is char, read from
                             SI1279.WRD file -> "bricks are an alternative"
                             Else if trans_type is phn,
                             read from SI1279.PHN file ->
                             "sil b r ih sil k s aa r er n aa l
                             sil t er n ih sil t ih v sil" """)
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    rs = []
    if args.non_lang_syms is not None:
        with codecs.open(args.non_lang_syms, 'r', encoding="utf-8") as f:
            nls = [x.rstrip() for x in f.readlines()]
            rs = [re.compile(re.escape(x)) for x in nls]

    if args.bpe_model is not None:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.load(args.bpe_model)

    if args.text:
        f = codecs.open(args.text, encoding="utf-8")
    else:
        f = codecs.getreader("utf-8")(
            sys.stdin if is_python2 else sys.stdin.buffer)

    sys.stdout = codecs.getwriter("utf-8")(
        sys.stdout if is_python2 else sys.stdout.buffer)
    line = f.readline()
    n = args.nchar
    while line:
        x = line.split()
        print(' '.join(x[:args.skip_ncols]), end=" ")
        a = ' '.join(x[args.skip_ncols:])

        # get all matched positions
        match_pos = []
        for r in rs:
            i = 0
            while i >= 0:
                m = r.search(a, i)
                if m:
                    match_pos.append([m.start(), m.end()])
                    i = m.end()
                else:
                    break

        if len(match_pos) > 0:
            chars = []
            i = 0
            while i < len(a):
                start_pos, end_pos = exist_or_not(i, match_pos)
                if start_pos is not None:
                    chars.append(a[start_pos:end_pos])
                    i = end_pos
                else:
                    chars.append(a[i])
                    i += 1
            a = chars

        if (args.trans_type == "phn"):
            a = a.split(" ")
        elif args.trans_type == "cn_char_en_bpe":
            b = seg_char(a)
            a = []
            for j in b:
                # we use "▁" to instead of blanks among english words
                # warning: here is "▁", not "_"
                for l in j.strip().split("▁"):
                    if not l.encode('UTF-8').isalpha():
                        a.append(l)
                    else:
                        for k in sp.encode_as_pieces(l):
                            a.append(k)
        else:
            a = [a[j:j + n] for j in range(0, len(a), n)]

        a_flat = []
        for z in a:
            a_flat.append("".join(z))

        a_chars = [z.replace(' ', args.space) for z in a_flat]
        if (args.trans_type == "phn"):
            a_chars = [z.replace("sil", args.space) for z in a_chars]
        print(' '.join(a_chars))
        line = f.readline()


if __name__ == '__main__':
    main()
