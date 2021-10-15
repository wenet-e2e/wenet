# Performance Record

## Conformer Result (Old IO)

* Feature info: using fbank feature, with cmvn, with speed perturb.
* Training info: lr 0.002, batch size 16, 1 machines, 1*4 = 4 gpu, acc_grad 4, 240 epochs, dither 0.1
* Decoding info: ctc_weight 0.5, average_num 30

| decoding mode            |       |
|--------------------------|-------|
| attention decoder        | 21.9  |
| ctc greedy search        | 21.15 |
| ctc prefix beam search   | 21.13 |
| attention rescoring      | 20.47 |

## Conformer Result (New IO)

* Feature info: using fbank feature, with cmvn, with speed perturb.
* Training info: lr 0.002, batch size 16, 1 machines, 1*4 = 4 gpu, acc_grad 4, 133 epochs, dither 0.1
* Decoding info: ctc_weight 0.5, average_num 30

| decoding mode            |       |
|--------------------------|-------|
| attention decoder        | 21.42 |
| ctc greedy search        | 21.16 |
| ctc prefix beam search   | 21.18 |
| attention rescoring      | 20.42 |
