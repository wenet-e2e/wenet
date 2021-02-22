# Performance Record

## Conformer Result

* Feature info: using fbank feature, cmvn, dither
* Training info: train_conformer.yaml, kernel size 31, lr 0.004, batch size 12, 8 gpu, acc_grad 4, 120 epochs, dither 0.1
* Decoding info: ctc_weight 0.5, average_num 30
* Git hash: 90d9a559840e765e82119ab72a11a1f7c1a01b78
* Model link: http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/librispeech/20210216_conformer_exp.tar.gz

| decoding mode            | test clean | test other |
|--------------------------|------------|------------|
| ctc greedy search        | 3.51       | 9.57       |
| ctc prefix beam search   | 3.51       | 9.56       |
| attention decoder        | 3.05       | 8.72       |
| attention rescoring      | 3.18       | 8.36       |


## Conformer U2 Result

* Feature info: using fbank feature, cmvn, speed perturb, dither
* Training info: train_unified_conformer.yaml lr 0.001, batch size 10, 8 gpu, acc_grad 1, 120 epochs, dither 1.0
* Decoding info: ctc_weight 0.5, average_num 30
* Git hash: 90d9a559840e765e82119ab72a11a1f7c1a01b78
* Model link: http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/librispeech/20210215_unified_conformer_exp.tar.gz

test clean
| decoding mode            | full | 16   |
|--------------------------|------|------|
| ctc prefix beam search   | 4.26 | 5.00 |
| attention decoder        | 3.05 | 3.44 |
| attention rescoring      | 3.72 | 4.10 |

test other
| decoding mode            | full  | 16    |
|--------------------------|-------|-------|
| ctc prefix beam search   | 10.87 | 12.87 |
| attention decoder        | 9.07  | 10.44 |
| attention rescoring      | 9.74  | 11.61 |
