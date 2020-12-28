# Performance Record

## Standard E2E

On conformer
* configure: conf/train_conformer.yaml
* beam size: 10
* gpu nums: 8
* ctc weight(used for attention rescoring): 0.5

| decoding mode/chunk size | full |
|--------------------------|------|
| attention decoder        | 4.97 |
| ctc greedy search        | 4.93 |
| ctc prefix beam search   | 4.93 |
| attention rescoring      | 4.70 |

On transformer
* configure: conf/train_transformer.yaml
* beam size: 10
* gpu nums: 8
* ctc weight(used for attention rescoring): 0.5

| decoding mode/chunk size | full |
|--------------------------|------|
| attention decoder        | 5.60 |
| ctc greedy search        | 5.92 |
| ctc prefix beam search   | 5.91 |
| attention rescoring      | 5.42 |



## Unified Dynamic chunk

On conformer(causal convolution)
* configure: conf/train_unified_conformer.yaml
* beam size: 10
* gpu nums: 8
* ctc weight(used for attention rescoring): 0.5

| decoding mode/chunk size | full | 16   | 8    | 4    | 1    |
|--------------------------|------|------|------|------|------|
| attention decoder        | 5.27 | 5.51 | 5.67 | 5.72 | 5.88 |
| ctc greedy search        | 5.49 | 6.08 | 6.41 | 6.64 | 7.58 |
| ctc prefix beam search   | 5.49 | 6.08 | 6.41 | 6.64 | 7.58 |
| attention rescoring      | 4.90 | 5.33 | 5.52 | 5.71 | 6.23 |

