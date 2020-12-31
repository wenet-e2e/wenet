# ASR Benchmark Results on LibriSpeech

## Standard E2E Results

Conformer without speed perpurb and lm
* config: conf/train_conformer_large.yaml
* beam: 10
* num of gpu: 8
* num of models average: 20
* ctc weight (used for attention rescoring): 0.5

test clean (chunk size = full)
| decoding mode            | WER  |
|--------------------------|------|
| attention rescoring      | 2.85 |

test other (chunk size = full)
| decoding mode            | WER  |
|--------------------------|------|
| attention rescoring      | 7.24 |

