# Performance Record

## Conformer Result

* Feature info: using fbank feature, dither=0, cmvn, speed perturb
* Training info: lr 0.004, batch size 12, 8 gpu, acc_grad 1, 70 epochs, dither 0.0
* Decoding info: ctc_weight 0.5, average_num 10
* Git hash: 1ea7515e4722f4e05a5985f587120c994a93694a
* Model link: http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/librispeech/20210122_conformer_exp.tar.gz

test clean (chunk size = full)
| decoding mode            | WER  |
|--------------------------|------|
| attention rescoring      | 3.53 |

test other (chunk size = full)
| decoding mode            | WER  |
|--------------------------|------|
| attention rescoring      | 9.69 |
