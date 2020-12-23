# Performance Record

## Standard E2E

On conformer without speed perpurb and lm
* configure: conf/train_conformer_large.yaml
* beam size: 10
* gpu nums: 8
* model averaged nums: 20
* ctc weight(used for attention rescoring): 0.5

test clean
| decoding mode/chunk size | full |
|--------------------------|------|
| attention rescoring      | 2.85 |

test other

| decoding mode/chunk size | full |
|--------------------------|------|
| attention rescoring      | 7.24 |

