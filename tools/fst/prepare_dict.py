#!/usr/bin/env python3
# encoding: utf-8

import sys

# sys.argv[1]: e2e model unit file(lang_char.txt)
# sys.argv[2]: raw lexicon file
# sys.argv[3]: output lexicon file

unit_table = set()
with open(sys.argv[1], 'r', encoding='utf8') as fin:
    for line in fin:
        unit = line.split()[0]
        unit_table.add(unit)

lexicon_table = set()
with open(sys.argv[2], 'r', encoding='utf8') as fin, \
     open(sys.argv[3], 'w', encoding='utf8') as fout:
    for line in fin:
        word = line.split()[0]
        if word == 'SIL':
            continue
        elif word == '<SPOKEN_NOISE>':
            continue
        else:
            # each word only has one pronunciation for e2e system
            if word in lexicon_table:
                continue
            # ignore words with OOV
            contains_oov = False
            for ch in word:
                if ch not in unit_table:
                    contains_oov = True
                    break
            if contains_oov:
                print(
                    'Ignoring words {}, which contains oov unit'.format(word))
                continue
            # Optional, append ▁ in front of english word
            # we assume the model unit of our e2e system is char now.
            # TODO(Binbin Zhang): support BPE by sentencepiece
            if word.encode('utf8').isalpha() and '▁' in unit_table:
                word = '▁' + word
            chars = ' '.join(word)  # word is a char list
            fout.write('{} {}\n'.format(word, chars))
            lexicon_table.add(word)
