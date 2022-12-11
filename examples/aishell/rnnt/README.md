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

## Conformer Result

* Feature info: using fbank feature, dither, cmvn, online speed perturb
* Training info: lr 0.001, batch size 20, 8 gpu, acc_grad 1, 140 epochs, dither 0.1
* Training weight info: transducer_weight 0.4, ctc_weight 0.2, attention_weight 0.4, average_num 10
* Predictor type: lstm
* Model link: https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/aishell/20220728_conformer_rnnt_exp.tar.gz

| decoding mode                         | CER   |
|---------------------------------------|-------|
| rnnt greedy search                    | 4.88  |
| rnnt beam search                      | 4.67  |
| ctc prefix beam search                | 5.02  |
| ctc prefix beam + rescore             | 4.51  |
| ctc prefix beam + rnnt&attn rescore   | 4.45  |
| rnnt prefix beam + rnnt&attn rescore  | 4.49  |


## U2++ Conformer Result

* Feature info: using fbank feature, dither, cmvn, oneline speed perturb
* Training info: lr 0.001, batch size 4, 32 gpu, acc_grad 1, 360 epochs
* Training weight info: transducer_weight 0.75,  ctc_weight 0.1, reverse_weight 0.15  average_num 30
* Predictor type: lstm

| decoding mode/chunk size  | full  | 16    |
|---------------------------|-------|-------|
| rnnt greedy search        | 5.68  | 6.26  |

## Pretrain
* Pretrain model: https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/aishell/20210601_u2%2B%2B_conformer_exp.tar.gz
* Feature info: using fbank feature, dither, cmvn, oneline speed perturb
* Training info: lr 0.001, batch size 8, 8 gpu, acc_grad 1, 140 epochs
* Training weight info: transducer_weight 0.4,  ctc_weight 0.2 , attention_weight 0.4, reverse_weight 0.3  average_num 30
* Predictor type: lstm

| decoding mode/chunk size    | full  | 16     |
|-----------------------------|-------|--------|
| rnnt greedy search          | 5.21  | 5.73   |
| rnnt prefix beam            | 5.14  | 5.63   |
| rnnt prefix beam + rescore  | 4.73  | 5.095  |


## Training loss ablation study

note:

- If rnnt is checked, greedy means rnnt  greedy search; so is beam

- if rnnt is checked, rescoring means rnnt beam & attention rescoring

- if only 'ctc & att' is checked, greedy means ctc gredy search; so is beam

- if only  'ctc & att' (AED)  is checked, rescoring means ctc beam & attention rescoring

- what if rnnt model do search of wenet's style, comming soon

| rnnt | ctc | att | greedy | beam | rescoring | fusion |
|------|-----|-----|--------|------|-----------|--------|
| ✔    | ✔   | ✔   |   4.88 | 4.67 |      4.45 |   4.49 |
| ✔    | ✔   |     |   5.56 | 5.46 |       /   |   5.40 |
| ✔    |     | ✔   |   5.03 | 4.94 |      4.87 |    /   |
| ✔    |     |     |   5.64 | 5.59 |       /   |    /   |
|      | ✔   | ✔   |   4.94 | 4.94 |      4.61 |    /   |
