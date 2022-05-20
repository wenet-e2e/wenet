# Performance Record

This is a Chinese speech recognition recipe that trains on all Chinese corpora including:

| Dataset    | Duration (Hours) |
|------------|------------------|
| Aidatatang | 140              |
| Aishell    | 151              |
| MagicData  | 712              |
| Primewords | 99               |
| ST-CMDS    | 110              |
| THCHS-30   | 26               |
| TAL-ASR    | 587              |
| AISHELL2   | 1000             |

## Unified Transformer Result

### Data info:

* Dataset: Aidatatang, Aishell, MagicData, Primewords, ST-CMDS, and THCHS-30.
* Feature info: using fbank feature, with cmvn, no speed perturb.
* Training info: lr 0.004, batch size 18, 3 machines, 3*8 = 24 GPUs, acc_grad 1, 220 epochs, dither 0.1
* Decoding info: ctc_weight 0.5, average_num 30
* Git hash: 013794572a55c7d0dbea23a66106ccf3e5d3b8d4

### WER

| Dataset    | chunk size | attention decoder | ctc greedy search | ctc prefix beam search | attention rescoring |
|------------|------------|-------------------|-------------------|------------------------|---------------------|
| Aidatatang | full       | 4.23              | 5.82              | 5.82                   | 4.71                |
|            | 16         | 4.59              | 6.99              | 6.99                   | 5.29                |
| Aishell    | full       | 4.69              | 5.80              | 5.80                   | 4.64                |
|            | 16         | 4.97              | 6.75              | 6.75                   | 5.37                |
| MagicData  | full       | 2.86              | 4.01              | 4.00                   | 3.07                |
|            | 16         | 3.10              | 5.02              | 5.02                   | 3.68                |
| THCHS-30   | full       | 16.68             | 15.46             | 15.46                  | 14.38               |
|            | 16         | 17.47             | 16.81             | 16.82                  | 15.63               |

## Unified Conformer Result

### Data info:

* Dataset: Aidatatang, Aishell, MagicData, Primewords, ST-CMDS, and THCHS-30.
* Feature info: using fbank feature, with cmvn, speed perturb.
* Training info: lr 0.001, batch size 8, 1 machines, 1*8 = 8 GPUs, acc_grad 12, 60 epochs
* Decoding info: ctc_weight 0.5, average_num 10
* Git hash: 5bdf436e671ef4c696d1b039f29cc33109e072fa

### WER

| Dataset    | chunk size | attention decoder | ctc greedy search | ctc prefix beam search | attention rescoring |
|------------|------------|-------------------|-------------------|------------------------|---------------------|
| Aidatatang | full       | 4.12              | 4.97              | 4.97                   | 4.22                |
|            | 16         | 4.45              | 5.73              | 5.73                   | 4.75                |
| Aishell    | full       | 4.49              | 5.07              | 5.05                   | 4.43                |
|            | 16         | 4.77              | 5.77              | 5.77                   | 4.85                |
| MagicData  | full       | 2.55              | 3.07              | 3.05                   | 2.59                |
|            | 16         | 2.81              | 3.88              | 3.86                   | 3.08                |
| THCHS-30   | full       | 13.55             | 13.75             | 13.76                  | 12.72               |
|            | 16         | 13.78             | 15.10             | 15.08                  | 13.90               |

## Unified Conformer Result

### Data info:

* Dataset: Aidatatang, Aishell, MagicData, Primewords, ST-CMDS, THCHS-30, TAL-ASR, and AISHELL2.
* Feature info: using fbank feature, dither=0, cmvn, speed perturb
* Training info: lr 0.001, batch size 22, 4 GPUs, acc_grad 4, 120 epochs, dither 0.1
* Decoding info: ctc_weight 0.5, average_num 10
* Git hash: 66f30c197d00c59fdeda3bc8ada801f867b73f78

### WER

| Dataset    | chunk size | attention decoder | ctc greedy search | ctc prefix beam search | attention rescoring |
|------------|------------|-------------------|-------------------|------------------------|---------------------|
| Aidatatang | full       | 3.22              | 4.00              | 4.01                   | 3.35                |
|            | 16         | 3.50              | 4.63              | 4.63                   | 3.79                |
| Aishell    | full       | 1.23              | 2.12              | 2.13                   | 1.42                |
|            | 16         | 1.33              | 2.72              | 2.72                   | 1.72                |
| MagicData  | full       | 2.38              | 3.07              | 3.05                   | 2.52                |
|            | 16         | 2.66              | 3.80              | 3.78                   | 2.94                |
| THCHS-30   | full       | 9.93              | 11.07             | 11.06                  | 10.16               |
|            | 16         | 10.28             | 11.85             | 11.85                  | 10.81               |
| AISHELL2   | full       | 5.25              | 5.81              | 5.79                   | 5.22                |
|            | 16         | 5.48              | 6.48              | 6.50                   | 5.61                |
| TAL-ASR    | full       | 9.54              | 10.35             | 10.28                  | 9.66                |
|            | 16         | 10.04             | 11.43             | 11.39                  | 10.55               |
