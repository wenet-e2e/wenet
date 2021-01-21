# Performance Record

## Conformer Result

* Feature info: using fbank feature, dither=0, no cmvn, no speed perturb.
* Training info: lr 0.001, batch size 16, 8 gpu, acc_grad 1, 80 epochs, dither 0.0
* Git hash: 9da70d74c444a64644f060a29766e0b7a1327719
* Model link: http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/aishell/20210115_conformer_exp.tar.gz

| decoding mode          | CER  |
|------------------------|------|
| attention decoder      | 5.95 |
| ctc greedy search      | 5.89 |
| ctc prefix beam search | 5.89 |
| attention rescoring    | 5.28 |

## Transformer Result

* Feature info: using fbank feature, dither=0, with cmvn, no speed perturb.
* Training info: lr 0.002, batch size 16, 8 gpu, acc_grad 1, 120 epochs, dither 0.0
* Git hash: fb8e0f8c12b5d547fc22e62365e1e114f059c609
* Model link: http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/aishell/20210120_transformer_exp.tar.gz

| decoding mode          | CER  |
|------------------------|------|
| attention decoder      | 5.76 |
| ctc greedy search      | 6.21 |
| ctc prefix beam search | 6.21 |
| attention rescoring    | 5.47 |

## Unified Transformer Result

* Feature info: using fbank feature, dither=0, with cmvn, no speed perturb.
* Training info: lr 0.002, batch size 16, 8 gpu, acc_grad 1, 120 epochs, dither 0.0
* Git hash: 2ba5394cb8ec0463271214d0c47c887e7b8128a0
* Model link: http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/aishell/20210121_unified_transformer_exp.tar.gz

| decoding mode/chunk size | full | 16   | 8    | 4     |
|--------------------------|------|------|------|-------|
| attention decoder        | 6.23 | 6.54 | 6.74 | 6.97  |
| ctc greedy search        | 7.05 | 8.58 | 9.56 | 11.32 |
| ctc prefix beam search   | 7.05 | 8.57 | 9.55 | 11.32 |
| attention rescoring      | 6.05 | 6.93 | 7.50 | 8.53  |

