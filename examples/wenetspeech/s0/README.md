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
* Decoding info: ctc_weight 0.5, reverse_weight 0.0, average_num 10, blank penalty 2.5, length penalty 8.5 for dev/testmeeting and 0.0 for testnet

| Decoding mode - Chunk size    | Dev  | Test\_Net | Test\_Meeting |
|:-----------------------------:|:----:|:---------:|:-------------:|
| ctc prefix beam search - full      | 7.21 % N=328207 C=309358 S=14175 D=4674 I=4801 | 9.46 % N=414285 C=381373 S=26013 D=6899 I=6295 | 14.02 % N=220358 C=195224 S=17266 D=7868 I=5754 |
| ctc prefix beam search - 16        | 7.93 % N=328207 C=307192 S=16529 D=4486 I=5000 | 11.14 % N=414285 C=374733 S=30241 D=9311 I=6596 | 16.37 % N=220358 C=191394 S=22435 D=6529 I=7116 |
| attention rescoring - full    | 7.10 % N=328207 C=308457 S=13215 D=6535 I=3537 | 8.83 % N=414285 C=381936 S=24808 D=7541 I=4215 | 13.64 % N=220358 C=194438 S=16238 D=9682 I=4133 |
| attention rescoring - 16      | 7.57 % N=328207 C=307065 S=15169 D=5973 I=3687 | 10.13 % N=414285 C=376854 S=28486 D=8945 I=4541 | 15.55 % N=220358 C=191270 S=21136 D=7952 I=5184 |
| attention - full    | 7.73 % N=328207 C=306688 S=13166 D=8353 I=3845 | 9.44 % N=414285 C=378096 S=24532 D=11657 I=2908 | 14.98 % N=220358 C=191881 S=15303 D=13174 I=4540 |

## U2++ conformer (text\_fixed, see https://github.com/wenet-e2e/WenetSpeech/discussions/54)

* Feature info: using fbank feature, with dither 1.0, with cmvn
* Training info: lr 0.001, batch size dynamic36000, 8 gpus on 3090, acc_grad 4, 130k steps, 4.6 days
* Decoding info: ctc_weight 0.5, reverse_weight 0.0, average_num 5, blank penalty 0.0, length penalty 0.0
* PR link: https://github.com/wenet-e2e/wenet/pull/2371

| Decoding mode - Chunk size    | Dev  | Test\_Net | Test\_Meeting |
|:-----------------------------:|:----:|:---------:|:-------------:|
| ctc prefix beam search - full      | 6.26 % N=328207 C=310671 S=15612 D=1924 I=3002 | 9.46 % N=414285 C=381373 S=26013 D=6899 I=6295 | 12.52 % N=220358 C=194801 S=19209 D=6348 I=2042 |
| attention rescoring - full    | 5.90 % N=328207 C=311721 S=14597 D=1889 I=2888 | 8.96 % N=414092 C=380232 S=27606 D=6254 I=3222 | 11.99 % N=220358 C=195808 S=18243 D=6307 I=1878 |
| attention - full    | 5.87 % N=328207 C=311922 S=14204 D=2081 I=2987 | 8.87 % N=414092 C=381014 S=27359 D=5719 I=3650 | 11.79 % N=220358 C=196484 S=17378 D=6496 I=2108 |

## U2++ conformer (wenetspeech plus aishell4)

* Feature info: using fbank feature, with dither 1.0, with cmvn
* Training info: lr 0.001, batch size dynamic36000, gradient checkpointing, torch_ddp, 8 * 3090 gpus, acc_grad 4, 60 epochs, about 8.5 days
* Decoding info: ctc_weight 0.5, reverse_weight 0.0, average_num 10, blank penalty 2.5 for dev and 0.0 for others

| Decoding mode - Chunk size    | Dev  | Test\_Net | Test\_Meeting |
|:-----------------------------:|:----:|:---------:|:-------------:|
| ctc prefix beam search - full      | 8.01 % N=328207 C=307477 S=15151 D=5579 I=5558 | 10.14 % N=414285 C=375271 S=27474 D=11540 I=2983 | 9.76 % N=220358 C=201205 S=13883 D=5270 I=2348 |
| attention rescoring - full    | 7.89 % N=328207 C=306307 S=13929 D=7971 I=3984 | 9.67 % N=414285 C=377058 S=25921 D=11306 I=2828 | 9.38 % N=220358 C=201833 S=13209 D=5316 I=2138 |
