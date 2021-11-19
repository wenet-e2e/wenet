# ASR Benchmark Results on AISHELL-1

## Standard E2E Results

Conformer
* feature: fbank
* config: conf/train_conformer.yaml
* beam: 10
* num of gpu: 4
* learning rate: 0.002
* ctc weight (used for attention rescoring): 0.8
* num of averaged model: 20
* use spec substitute

| decoding mode/chunk size | full |
|--------------------------|------|
| attention decoder        | 5.15 |
| ctc greedy search        | 4.85 |
| ctc prefix beam search   | 4.85 |
| attention rescoring      | 4.48 |

Conformer
* feature: fbank
* config: conf/train_conformer.yaml
* beam: 10
* num of gpu: 4
* learning rate: 0.002
* ctc weight (used for attention rescoring): 0.6
* num of averaged model: 20

| decoding mode/chunk size | full |
|--------------------------|------|
| attention decoder        | 5.20 |
| ctc greedy search        | 4.92 |
| ctc prefix beam search   | 4.92 |
| attention rescoring      | 4.61 |

Conformer
* feature: fbank & pitch
* config: conf/train_conformer.yaml
* beam: 10
* num of gpu: 4
* learning rate: 0.002
* ctc weight (used for attention rescoring): 0.7
* num of averaged model: 20

| decoding mode/chunk size | full |
|--------------------------|------|
| attention decoder        | 4.92 |
| ctc greedy search        | 4.93 |
| ctc prefix beam search   | 4.93 |
| attention rescoring      | 4.64 |


Transformer
* config: conf/train_transformer.yaml
* beam: 10
* num of gpu: 8
* ctc weight (used for attention rescoring): 0.5

| decoding mode/chunk size | full |
|--------------------------|------|
| attention decoder        | 5.67 |
| ctc greedy search        | 5.88 |
| ctc prefix beam search   | 5.88 |
| attention rescoring      | 5.30 |



## Unified Dynamic Chunk Results

Conformer (causal convolution)
* config: conf/train_unified_conformer.yaml
* beam: 10
* num of gpu: 8
* ctc weight (used for attention rescoring): 0.5

| decoding mode/chunk size | full | 16   | 8    | 4    | 1    |
|--------------------------|------|------|------|------|------|
| attention decoder        | 5.27 | 5.51 | 5.67 | 5.72 | 5.88 |
| ctc greedy search        | 5.49 | 6.08 | 6.41 | 6.64 | 7.58 |
| ctc prefix beam search   | 5.49 | 6.08 | 6.41 | 6.64 | 7.58 |
| attention rescoring      | 4.90 | 5.33 | 5.52 | 5.71 | 6.23 |

