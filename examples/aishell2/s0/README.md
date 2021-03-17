# Performance Record

## Unified Transformer Result

* Feature info: using fbank feature, with cmvn, no speed perturb.
* Training info: lr 0.004, batch size 18, 3 machines, 3*8 = 24 gpu, acc_grad 1, 120 epochs, dither 0.1
* Decoding info: ctc_weight 0.5, average_num 30
* Git hash: 18e3bd2448f6b73bc39bac2e1c5a76280b847ca3
* Model link: http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/aishell2/20210317_unified_transformer_exp.tar.gz

| decoding mode/chunk size | full | 16   | 8     | 4     |
|--------------------------|------|------|-------|-------|
| attention decoder        | 7.02 | 7.44 | 7.59  | 7.81  |
| ctc greedy search        | 8.38 | 9.67 | 10.30 | 11.02 |
| ctc prefix beam search   | 8.40 | 9.67 | 10.29 | 11.02 |
| attention rescoring      | 7.04 | 7.79 | 8.18  | 8.71  |
