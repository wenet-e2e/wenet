#!/usr/bin/env python3
import os
import argparse

conversational_filler = [
    'UH', 'UHH', 'UM', 'EH', 'MM', 'HM', 'AH', 'HUH', 'HA', 'ER', 'OOF', 'HEE',
    'ACH', 'EEE', 'EW'
]
unk_tags = ['<UNK>', '<unk>']
gigaspeech_punctuations = [
    '<COMMA>', '<PERIOD>', '<QUESTIONMARK>', '<EXCLAMATIONPOINT>'
]
gigaspeech_garbage_utterance_tags = ['<SIL>', '<NOISE>', '<MUSIC>', '<OTHER>']
non_scoring_words = conversational_filler + unk_tags + \
    gigaspeech_punctuations + gigaspeech_garbage_utterance_tags

def asr_text_post_processing(text):
    # 1. convert to uppercase
    text = text.upper()

    # 2. remove hyphen
    #   "E-COMMERCE" -> "E COMMERCE", "STATE-OF-THE-ART" -> "STATE OF THE ART"
    text = text.replace('-', ' ')

    # 3. remove non-scoring words from evaluation
    remaining_words = []
    for word in text.split():
        if word in non_scoring_words:
            continue
        remaining_words.append(word)

    return ' '.join(remaining_words)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''This script evaluates GigaSpeech ASR
                     result via SCTK's tool sclite''')
    parser.add_argument(
        'ref',
        type=str,
        help="sclite's standard transcription(trn) reference file")
    parser.add_argument(
        'hyp',
        type=str,
        help="sclite's standard transcription(trn) hypothesis file")
    parser.add_argument('work_dir', type=str, help='working dir')
    args = parser.parse_args()

    if not os.path.isdir(args.work_dir):
        os.mkdir(args.work_dir)

    REF = os.path.join(args.work_dir, 'REF')
    HYP = os.path.join(args.work_dir, 'HYP')
    RESULT = os.path.join(args.work_dir, 'RESULT')

    for io in [(args.ref, REF), (args.hyp, HYP)]:
        with open(io[0],
                  'r', encoding='utf8') as fi, open(io[1],
                                                    'w+',
                                                    encoding='utf8') as fo:
            for line in fi:
                line = line.strip()
                if line:
                    cols = line.split()
                    text = asr_text_post_processing(' '.join(cols[0:-1]))
                    uttid_field = cols[-1]
                    print(F'{text} {uttid_field}', file=fo)

    os.system(F'sclite -r {REF} trn -h {HYP} trn -i swb | tee {RESULT}'
              )  # GigaSpeech's uttid comforms to swb
