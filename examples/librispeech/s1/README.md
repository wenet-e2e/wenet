# ASR Benchmark Results on LibriSpeech

## Standard E2E Results

Conformer without speed perpurb and lm
* config: conf/train_conformer_large.yaml
* beam: 10
* num of gpu: 8
* num of averaged model: 20
* ctc weight (used for attention rescoring): 0.5

test clean (chunk size = full)
| decoding mode            | WER  |
|--------------------------|------|
| attention rescoring      | 2.85 |

test other (chunk size = full)
| decoding mode            | WER  |
|--------------------------|------|
| attention rescoring      | 7.24 |


## Unified Dynamic Chunk Results

Conformer (causal convolution)
* config: conf/train_unified_conformer.yaml
* beam: 10
* num of gpu: 8
* ctc weight (used for attention rescoring): 0.5
* num of averaged model: 30

test clean
| decoding mode/chunk size | full | 16   |
|--------------------------|------|------|
| attention decoder        | 5.17 | 5.21 |
| ctc greedy search        | 3.99 | 4.74 |
| ctc prefix beam search   | 3.99 | 4.74 |
| attention rescoring      | 3.39 | 3.94 |

test other
| decoding mode/chunk size | full | 16    |
|--------------------------|------|-------|
| attention decoder        | 9.41 | 10.75 |
| ctc greedy search        | 9.80 | 11.86 |
| ctc prefix beam search   | 9.80 | 11.85 |
| attention rescoring      | 8.64 | 10.52 |
