# Performance Record

## Conformer Result

* Feature info: using fbank feature, dither=0, cmvn, speed perturb
* Training info: lr 0.004, batch size 12, 8 gpu, acc_grad 1, 70 epochs, dither 0.0
* Decoding info: ctc_weight 0.5, average_num 10

test clean (chunk size = full)
| decoding mode            | WER  |
|--------------------------|------|
| attention rescoring      | 3.59 |

test other (chunk size = full)
| decoding mode            | WER  |
|--------------------------|------|
| attention rescoring      | 9.85 |
