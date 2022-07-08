# Performance Record

## Conformer Result

* Feature info: using fbank feature, dither, cmvn, online speed perturb
* Training info: lr 0.001, batch size 8, 8 gpu, acc_grad 1, 100 epochs, dither 0.1
* Training weight info: transducer_weight 0.75, ctc_weight 0.1, attention_weight 0.15, average_num 10
* Predictor type: lstm

| decoding mode             | CER   |
|---------------------------|-------|
| rnnt greedy search        | 5.24  |

* after 165 epochs and avg 30

| decoding mode             | CER   |
|---------------------------|-------|
| rnnt greedy search        | 5.02  |
| ctc prefix beam search    | 5.17  |
| ctc prefix beam + rescore | 4.48  |

## conformer Result

* Feature info: using fbank feature, dither, cmvn, online speed perturb
* Training info: lr 0.001, batch size 20, 8 gpu, acc_grad 1, 140 epochs, dither 0.1
* Training weight info: transducer_weight 0.4, ctc_weight 0.2, attention_weight 0.4, average_num 10
* Predictor type: lstm

| decoding mode             | CER   |
|---------------------------|-------|
| rnnt greedy search        | 4.88  |
| ctc prefix beam search    | 5.02  |
| ctc prefix beam + rescore | 4.51  |


## U2++ Conformer Result

* Feature info: using fbank feature, dither, cmvn, oneline speed perturb
* Training info: lr 0.001, batch size 4, 32 gpu, acc_grad 1, 360 epochs
* Training weight info: transducer_weight 0.75,  ctc_weight 0.1, reverse_weight 0.15  average_num 30
* Predictor type: lstm

| decoding mode/chunk size  | full  | 16    |
|---------------------------|-------|-------|
| rnnt greedy search        | 5.68  | 6.26  |
