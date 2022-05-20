# Performance Record

## U2++ Conformer Result

* Feature info: using fbank feature, with cmvn, no speed perturb, dither
* Training info: lr 0.001, batch size 32, 8 gpus, acc_grad 1, 240 epochs, dither 1.0
* Decoding info: ctc_weight 0.1, reverse_weight 0.4, average_num 30
* Git hash: 5a1342312668e7a5abb83aed1e53256819cebf95

| decoding mode/chunk size  | full  | 16    |
|---------------------------|-------|-------|
| ctc greedy search         | 6.18  | 6.79  |
| ctc prefix beam search    | 6.20  | 6.80  |
| attention rescoring       | 5.39  | 5.78  |
| LM + attention rescoring  | 5.35  | 5.73  |

## U2++ Transformer Result

* Feature info: using fbank feature, with cmvn, no speed perturb
* Training info: lr 0.002, batch size 22, 8 gpus, acc_grad 1, 240 epochs, dither 0.0
* Decoding info: ctc_weight 0.1, reverse_weight 0.5, average_num 30
* Git hash: 5a1342312668e7a5abb83aed1e53256819cebf95

| decoding mode/chunk size  | full  | 16    |
|---------------------------|-------|-------|
| ctc greedy search         | 7.35  | 8.23  |
| ctc prefix beam search    | 7.36  | 8.23  |
| attention rescoring       | 6.09  | 6.70  |
| LM + attention rescoring  | 6.07  | 6.55  |

## Unified Conformer Result

* Feature info: using fbank feature, with cmvn, no speed perturb.
* Training info: lr 0.002, batch size 16, 8 gpus, acc_grad 1, 120 epochs, dither 1.0
* Decoding info: ctc_weight 0.5, average_num 20
* Git hash: 14d38085a8d966cf9e9577ffafc51d578dce954f

| decoding mode/chunk size  | full  | 16    | 8     | 4     |
|---------------------------|-------|-------|-------|-------|
| attention decoder         | 6.23  | 6.42  | 6.58  | 7.20  |
| ctc greedy search         | 6.98  | 7.75  | 8.21  | 9.91  |
| ctc prefix beam search    | 7.02  | 7.76  | 8.21  | 9.93  |
| attention rescoring       | 6.08  | 6.46  | 6.72  | 7.79  |
| LM + attention rescoring  | 5.87  | 6.37  | 6.47  | 6.61  |

## Unified Transformer Result

* Feature info: using fbank feature, with cmvn, no speed perturb.
* Training info: lr 0.002, batch size 22, 8 gpus, acc_grad 1, 180 epochs, dither 0.0
* Decoding info: ctc_weight 0.5, average_num 30
* Git hash: 14d38085a8d966cf9e9577ffafc51d578dce954f

| decoding mode/chunk size  | full  | 16    | 8     | 4     |
|---------------------------|-------|-------|-------|-------|
| attention decoder         | 6.71  | 7.08  | 7.17  | 7.40  |
| ctc greedy search         | 7.84  | 8.68  | 8.98  | 9.46  |
| ctc prefix beam search    | 7.86  | 8.68  | 8.98  | 9.45  |
| attention rescoring       | 6.71  | 7.31  | 7.51  | 7.85  |
| LM + attention rescoring  | 6.35  | 7.02  | 7.24  | 7.52  |
