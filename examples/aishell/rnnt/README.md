# Performance Record

## Conformer Result

* Feature info: using fbank feature, dither, cmvn, online speed perturb
* Training info: lr 0.001, batch size 8, 8 gpu, acc_grad 1, 100 epochs, dither 0.1
* Decoding info: transducer_weight 0.75, ctc_weight 0.1, attention_weight 0.15, average_num 10

| decoding mode             | CER   |
|---------------------------|-------|
| rnnt greedy search        | 5.24  |

* after 165 epochs and avg 30

| decoding mode             | CER   |
|---------------------------|-------|
| rnnt greedy search        | 5.02  |

## U2++ Conformer Result

* Feature info: using fbank feature, dither, cmvn, oneline speed perturb
* Training info: lr 0.001, batch size 4, 32 gpu, acc_grad 1, 360 epochs
* Decoding info: transducer_weight 0.75,  ctc_weight 0.1, reverse_weight 0.15  average_num 30

| decoding mode/chunk size  | full  | 16    |
|---------------------------|-------|-------|
| rnnt greedy search        | 5.68  | 6.26  |
