# Performance Record

## Unified Transformer Result

* Feature info: using fbank feature, with cmvn, no speed perturb.
* Training info: lr 0.002, batch size 22, 8 gpus, acc_grad 1, 180 epochs, dither 0.0
* Decoding info: ctc_weight 0.5, average_num 30
* Git hash: 14d38085a8d966cf9e9577ffafc51d578dce954f
* Model link: http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/aishell2/20210421_unified_transformer_exp.tar.gz

| decoding mode/chunk size  | full  | 16    | 8     | 4     |
|---------------------------|-------|-------|-------|-------|
| attention decoder         | 6.71  | 7.08  | 7.17  | 7.40  |
| ctc greedy search         | 7.84  | 8.68  | 8.98  | 9.46  |
| ctc prefix beam search    | 7.86  | 8.68  | 8.98  | 9.45  |
| attention rescoring       | 6.71  | 7.31  | 7.51  | 7.85  |
| LM + attention rescoring  | 6.35  | 7.02  | 7.24  | 7.52  |

## Unified Conformer Result

* Feature info: using fbank feature, with cmvn, no speed perturb.
* Training info: lr 0.002, batch size 16, 8 gpus, acc_grad 1, 120 epochs, dither 1.0
* Decoding info: ctc_weight 0.5, average_num 20
* Git hash: 14d38085a8d966cf9e9577ffafc51d578dce954f
* Model link: http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/aishell2/20210421_unified_conformer_exp.tar.gz

| decoding mode/chunk size  | full  | 16    | 8     | 4     |
|---------------------------|-------|-------|-------|-------|
| attention decoder         | 6.23  | 6.42  | 6.58  | 7.20  |
| ctc greedy search         | 6.98  | 7.75  | 8.21  | 9.91  |
| ctc prefix beam search    | 7.02  | 7.76  | 8.21  | 9.93  |
| attention rescoring       | 6.08  | 6.46  | 6.72  | 7.79  |
| LM + attention rescoring  | 5.87  | 6.37  | 6.47  | 6.61  |
