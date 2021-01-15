# Performance Record

## Conformer Result

* Feature info: using fbank feature, dither=0, no cmvn, no speed perturb.
* Training info: lr 0.001, batch size 16, 8 gpu, acc_grad 1, 80 epochs, dither 0.0
* Git hash: (TODO Add here)
* Model link: (TODO Add here)

| decoding mode          | CER  |
|------------------------|------|
| attention decoder      | 5.95 |
| ctc greedy search      | 5.89 |
| ctc prefix beam search | 5.89 |
| attention rescoring    | 5.28 |


