# Performance Record

## Conformer Result

* Feature info: using fbank feature, dither, cmvn, without speed perturb (not supported segments yet)
* Training info: lr 0.001, batch size 20, 8 gpu, acc_grad 1, 240 epochs, dither 0.1
* Decoding info: ctc_weight 0.5, average_num 10


| decoding mode       | Dev WER | Test WER |
|---------------------|---------|----------|
| attention rescoring | 9.54%   | 8.66%    |