# ASR Benchmark Results on AISHELL-1

## Standard E2E Results

Conformer
* config: conf/train_conformer.yaml
* beam: 10
* num of gpu: 8
* ctc weight (used for attention rescoring): 0.5

| decoding mode/chunk size | full |
|--------------------------|------|
| attention decoder        | 4.97 |
| ctc greedy search        | 4.93 |
| ctc prefix beam search   | 4.93 |
| attention rescoring      | 4.70 |

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

