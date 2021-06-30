# Performance Record

This is a Chinese speech recognition recipe that trains on all Chinese corpora including:
* Aidatatang (140 hours)
* Aishell (151 hours)
* MagicData (712 hours)
* Primewords (99 hours)
* ST-CMDS (110 hours)
* THCHS-30 (26 hours)
* optional AISHELL2 (~1000 hours) if available
* optional TAL ASR (~600 hours) if available

## Unified Transformer Result
### data info:
* Aidatatang (140 hours)
* Aishell (151 hours)
* MagicData (712 hours)
* Primewords (99 hours)
* ST-CMDS (110 hours)
* THCHS-30 (26 hours)

* Feature info: using fbank feature, with cmvn, no speed perturb.
* Training info: lr 0.004, batch size 18, 3 machines, 3*8 = 24 gpu, acc_grad 1, 220 epochs, dither 0.1
* Decoding info: ctc_weight 0.5, average_num 30
* Git hash: 013794572a55c7d0dbea23a66106ccf3e5d3b8d4
* Model link: http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/multi_cn/20210315_unified_transformer_exp.tar.gz

aishell results

| decoding mode/chunk size | full | 16   |
|--------------------------|------|------|
| attention decoder        | 4.69 | 4.97 |
| ctc greedy search        | 5.80 | 6.75 |
| ctc prefix beam search   | 5.80 | 6.75 |
| attention rescoring      | 4.64 | 5.37 |

aidatatang results

| decoding mode/chunk size | full | 16   |
|--------------------------|------|------|
| attention decoder        | 4.23 | 4.59 |
| ctc greedy search        | 5.82 | 6.99 |
| ctc prefix beam search   | 5.82 | 6.99 |
| attention rescoring      | 4.71 | 5.29 |

thcs30 results

| decoding mode/chunk size | full  | 16    |
|--------------------------|-------|-------|
| attention decoder        | 16.68 | 17.47 |
| ctc greedy search        | 15.46 | 16.81 |
| ctc prefix beam search   | 15.46 | 16.82 |
| attention rescoring      | 14.38 | 15.63 |

magic results

| decoding mode/chunk size | full | 16   |
|--------------------------|------|------|
| attention decoder        | 2.86 | 3.10 |
| ctc greedy search        | 4.01 | 5.02 |
| ctc prefix beam search   | 4.00 | 5.02 |
| attention rescoring      | 3.07 | 3.68 |

## Unified Conformer Result
### data info:
* Aidatatang (140 hours)
* Aishell (151 hours)
* MagicData (712 hours)
* Primewords (99 hours)
* ST-CMDS (110 hours)
* THCHS-30 (26 hours)

* Feature info: using fbank feature, with cmvn, speed perturb.
* Training info: lr 0.001, batch size 8, 1 machines, 1*8 = 8 gpu, acc_grad 12, 60 epochs
* Decoding info: ctc_weight 0.5, average_num 10
* Git hash: 5bdf436e671ef4c696d1b039f29cc33109e072fa
* Model link:

aishell results

| decoding mode/chunk size | full | 16   |
|--------------------------|------|------|
| attention decoder        | 4.49 | 4.77 |
| ctc greedy search        | 5.07 | 5.77 |
| ctc prefix beam search   | 5.05 | 5.77 |
| attention rescoring      | 4.43 | 4.85 |

aidatatang results

| decoding mode/chunk size | full | 16   |
|--------------------------|------|------|
| attention decoder        | 4.12 | 4.45 |
| ctc greedy search        | 4.97 | 5.73 |
| ctc prefix beam search   | 4.97 | 5.73 |
| attention rescoring      | 4.22 | 4.75 |

thcs30 results

| decoding mode/chunk size | full  | 16    |
|--------------------------|-------|-------|
| attention decoder        | 13.55 | 13.78 |
| ctc greedy search        | 13.75 | 15.10 |
| ctc prefix beam search   | 13.76 | 15.08 |
| attention rescoring      | 12.72 | 13.90 |

magic results

| decoding mode/chunk size | full | 16   |
|--------------------------|------|------|
| attention decoder        | 2.55 | 2.81 |
| ctc greedy search        | 3.07 | 3.88 |
| ctc prefix beam search   | 3.05 | 3.86 |
| attention rescoring      | 2.59 | 3.08 |

## Unified Conformer Result

### data info:
* Aidatatang (140 hours)
* Aishell (151 hours)
* MagicData (712 hours)
* Primewords (99 hours)
* ST-CMDS (110 hours)
* THCHS-30 (26 hours)
* AISHELL2 (~1000 hours)
* TAL ASR (~600 hours)

* Feature info: using fbank feature, dither=0, cmvn, speed perturb
* Training info: lr 0.001, batch size 12, 16 gpu, acc_grad 4, 48 epochs, dither 0.1
* Decoding info: ctc_weight 0.5, average_num 10
* Git hash: 18a75244f39b3403ff7b39f791e7af4fb93d4d03
* Model link:

aishell results

| decoding mode/chunk size | full | 16   |
|--------------------------|------|------|
| attention decoder        | 1.65 | 1.79 |
| ctc greedy search        | 2.14 | 2.79 |
| ctc prefix beam search   | 2.13 | 2.79 |
| attention rescoring      | 1.65 | 2.01 |

aidatatang results

| decoding mode/chunk size | full | 16   |
|--------------------------|------|------|
| attention decoder        | 3.88 | 4.21 |
| ctc greedy search        | 5.39 | 5.88 |
| ctc prefix beam search   | 5.39 | 5.87 |
| attention rescoring      | 4.02 | 4.54 |

thcs30 results

| decoding mode/chunk size | full  | 16    |
|--------------------------|-------|-------|
| attention decoder        | 9.67  | 10.07 |
| ctc greedy search        | 10.94 | 11.95 |
| ctc prefix beam search   | 10.94 | 11.96 |
| attention rescoring      | 9.90  | 10.74 |

magic results

| decoding mode/chunk size | full | 16   |
|--------------------------|------|------|
| attention decoder        | 2.66 | 2.98 |
| ctc greedy search        | 3.18 | 3.96 |
| ctc prefix beam search   | 3.17 | 3.95 |
| attention rescoring      | 2.71 | 3.23 |

aishell-2 results

| decoding mode/chunk size | full | 16   |
|--------------------------|------|------|
| attention decoder        | 5.37 | 5.67 |
| ctc greedy search        | 5.87 | 6.56 |
| ctc prefix beam search   | 5.88 | 6.57 |
| attention rescoring      | 5.27 | 5.78 |

tal results

| decoding mode/chunk size | full | 16   |
|--------------------------|------|------|
| attention decoder        | 10.49 | 11.09 |
| ctc greedy search        | 11.11 | 12.14 |
| ctc prefix beam search   | 11.05 | 12.06 |
| attention rescoring      | 10.49 | 11.32 |
