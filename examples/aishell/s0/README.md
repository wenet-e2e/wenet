# Performance Record

## Conformer Result

* Feature info: using fbank feature, dither, cmvn, online speed perturb
* Training info: lr 0.002, batch size 18, 4 gpu, acc_grad 4, 240 epochs, dither 0.1
* Decoding info: ctc_weight 0.5, average_num 20
* Git hash: 919f07c4887ac500168ba84b39b535fd8e58918a
* Model link: http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/aishell/20210204_conformer_exp.tar.gz

| decoding mode          | CER  |
|------------------------|------|
| attention decoder      | 5.18 |
| ctc greedy search      | 4.94 |
| ctc prefix beam search | 4.94 |
| attention rescoring    | 4.61 |

## Unified Conformer Result

* Feature info: using fbank feature, dither=0, cmvn, oneline speed perturb
* Training info: lr 0.001, batch size 16, 8 gpu, acc_grad 1, 180 epochs, dither 0.0
* Decoding info: ctc_weight 0.5, average_num 20
* Git hash: 919f07c4887ac500168ba84b39b535fd8e58918a
* Model link: http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/aishell/20210203_unified_conformer_exp.tar.gz

| decoding mode/chunk size | full | 16   | 8    | 4    |
|--------------------------|------|------|------|------|
| attention decoder        | 5.40 | 5.60 | 5.74 | 5.86 |
| ctc greedy search        | 5.56 | 6.29 | 6.68 | 7.10 |
| ctc prefix beam search   | 5.57 | 6.30 | 6.67 | 7.10 |
| attention rescoring      | 5.05 | 5.45 | 5.69 | 5.91 |

## Transformer Result

* Feature info: using fbank feature, dither, with cmvn, no speed perturb.
* Training info: lr 0.002, batch size 26, 4 gpu, acc_grad 4, 240 epochs, dither 0.1
* Decoding info: ctc_weight 0.5, average_num 20
* Git hash: 919f07c4887ac500168ba84b39b535fd8e58918a
* Model link: http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/aishell/20210204_transformer_exp.tar.gz

| decoding mode          | CER  |
|------------------------|------|
| attention decoder      | 5.69 |
| ctc greedy search      | 5.92 |
| ctc prefix beam search | 5.91 |
| attention rescoring    | 5.30 |

## Unified Transformer Result

* Feature info: using fbank feature, dither=0, with cmvn, no speed perturb.
* Training info: lr 0.002, batch size 16, 8 gpu, acc_grad 1, 120 epochs, dither 0.0
* Decoding info: ctc_weight 0.5, average_num 20
* Git hash: 919f07c4887ac500168ba84b39b535fd8e58918a
* Model link: http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/aishell/20210204_unified_transformer_exp.tar.gz

| decoding mode/chunk size | full | 16   | 8    | 4    |
|--------------------------|------|------|------|------|
| attention decoder        | 6.13 | 6.43 | 6.55 | 6.79 |
| ctc greedy search        | 6.73 | 7.99 | 8.72 | 9.92 |
| ctc prefix beam search   | 6.73 | 7.99 | 8.73 | 9.91 |
| attention rescoring      | 5.80 | 6.56 | 7.02 | 7.68 |

## Muilti-machines Training Conformer Result

* Feature info: using fbank feature, dither, cmvn, online speed perturb
* Training info: lr 0.004, batch size 16, 2 machines, 8*2=16 gpus, acc_grad 4, 240 epochs, dither 0.1, warm up steps 10000
* Decoding info: ctc_weight 0.5, average_num 20
* Git hash: f6b1409023440da1998d31abbcc3826dd40aaf35

| decoding mode          | CER  |
|------------------------|------|
| attention decoder      | 4.90 |
| ctc greedy search      | 5.07 |
| ctc prefix beam search | 5.06 |
| attention rescoring    | 4.65 |
