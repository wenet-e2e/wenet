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
* Training info: lr 0.002, batch size dynamic24000, 24 gpus on 3090, acc_grad 16, 80 epochs, 4.5 days
* Decoding info: ctc_weight 0.5, reverse_weight 0.0, average_num 10

| Decoding mode - Chunk size    | Dev  | Test\_Net | Test\_Meeting |
|:-----------------------------:|:----:|:---------:|:-------------:|
| ctc greedy search - full      | 8.50 | 9.47      | 15.77         |
| ctc greedy search - 16        | 9.01 | 11.14     | 16.89         |
| attention rescoring - full    | 8.37 | 9.02      | 15.52         |
| attention rescoring - 16      | 8.62 | 10.30     | 16.32         |
