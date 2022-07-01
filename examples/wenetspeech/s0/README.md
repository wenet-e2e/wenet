# Performance Record

## Conformer

* Feature info: using fbank feature, with dither 1.0, with cmvn
* Training info: lr 0.001, batch size 32, 24 gpus on V100, acc_grad 16, 26 epochs
* Decoding info: ctc_weight 0.5, average_num 10

|   decoding_method   |  Dev | Test\_Net | Test\_Meeting |
|:-------------------:|:----:|:---------:|:-------------:|
|  ctc_greedy_search  | 8.88 |   10.29   |     15.96     |
|      attention      | 9.38 |   10.12   |     17.28     |
| attention_rescoring | 8.69 |    9.7    |     15.59     |

## Conformer bidecoder

* Feature info: using fbank feature, with dither 1.0, with cmvn
* Training info: lr 0.001, batch size 32, 24 gpus on V100, acc_grad 16, 26 epochs
* Decoding info: ctc_weight 0.5, average_num 10

|   decoding_method   |  Dev | Test\_Net | Test\_Meeting |
|:-------------------:|:----:|:---------:|:-------------:|
|  ctc_greedy_search  | 8.98 |    9.55   |     16.48     |
|      attention      | 9.42 |   10.57   |     18.05     |
| attention_rescoring | 8.85 |    9.25   |     16.18     |

## U2++ conformer

* Feature info: using fbank feature, with dither 1.0, with cmvn
* Training info: lr 0.001, batch size 48, 8 gpus on A100, acc_grad 16, 50 epochs
* Decoding info: ctc_weight 0.5, reverse_weight 0.3, average_num 10

| Decoding mode - Chunk size    | Dev  | Test\_Net | Test\_Meeting |
|:-----------------------------:|:----:|:---------:|:-------------:|
| ctc greedy search - full      | 8.85 | 9.78      | 17.77         |
| ctc greedy search - 16        | 9.32 | 11.02     | 18.79         |
| ctc prefix beam search - full | 8.80 | 9.73      | 17.57         |
| ctc prefix beam search - 16   | 9.25 | 10.96     | 18.62         |
| attention rescoring - full    | 8.60 | 9.26      | 17.34         |
| attention rescoring - 16      | 8.87 | 10.22     | 18.11         |
