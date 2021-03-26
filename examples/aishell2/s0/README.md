# Performance Record

## Unified Transformer Result

* Feature info: using fbank feature, with cmvn, no speed perturb.
* Training info: lr 0.002, batch size 22, 8 gpus, acc_grad 1, 130 epochs, dither 0.0
* Decoding info: ctc_weight 0.5, average_num 10
* Git hash: fa583b603c1d32023467389b2055fece2e399e88
* Model link: http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/aishell2/20210326_unified_transformer_exp.tar.gz

| decoding mode/chunk size | full | 16   | 8    | 4    |
|--------------------------|------|------|------|------|
| attention decoder        | 6.82 | 7.22 | 7.37 | 7.57 |
| ctc greedy search        | 8.02 | 8.90 | 9.30 | 9.79 |
| ctc prefix beam search   | 8.05 | 8.90 | 9.28 | 9.77 |
| attention rescoring      | 6.90 | 7.44 | 7.69 | 8.03 |
