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
* Decoding info: ctc_weight 0.5, reverse_weight 0.0, average_num 10, blank penalty 2.5

| Decoding mode - Chunk size    | Dev  | Test\_Net | Test\_Meeting |
|:-----------------------------:|:----:|:---------:|:-------------:|
| ctc prefix beam search - full      | 7.21 % N=328207 C=309358 S=14175 D=4674 I=4801 | 9.46 % N=414285 C=381373 S=26013 D=6899 I=6295 | 14.02 % N=220358 C=195224 S=17266 D=7868 I=5754 |
| ctc prefix beam search - 16        | 7.93 % N=328207 C=307192 S=16529 D=4486 I=5000 | 11.14 % N=414285 C=374733 S=30241 D=9311 I=6596 | 16.37 % N=220358 C=191394 S=22435 D=6529 I=7116 |
| attention rescoring - full    | 7.10 % N=328207 C=308457 S=13215 D=6535 I=3537 | 8.83 % N=414285 C=381936 S=24808 D=7541 I=4215 | 13.64 % N=220358 C=194438 S=16238 D=9682 I=4133 |
| attention rescoring - 16      | 7.57 % N=328207 C=307065 S=15169 D=5973 I=3687 | 10.13 % N=414285 C=376854 S=28486 D=8945 I=4541 | 15.55 % N=220358 C=191270 S=21136 D=7952 I=5184 |
