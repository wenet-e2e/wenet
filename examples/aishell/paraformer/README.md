# Performance Record


## Paraformer, Conformer Result

* Feature info: using fbank feature, dither, cmvn, online speed perturb
* Training info: lr 0.002, batch size 16, 4 gpu, acc_grad 4, 240 epochs, dither 0.1
* Decoding info: ctc_weight 0.5, average_num 20

| decoding mode          | Dev CER | Test CER |
|------------------------| --- | ---- |
| ctc greedy search      | 4.65 | 5.24 |
| ctc prefix beam search | 4.65 | 5.24 |
| cif greedy search      | 4.41 | 4.92 |
| cif beam search        | 4.35 | 4.86 |

## Paraformer, Conformer DecoderSANM Result(Deprecated)

* Feature info: using fbank feature, dither, cmvn, online speed perturb
* Training info: lr 0.002, batch size 16, 4 gpu, acc_grad 4, 240 epochs, dither 0.1
* Decoding info: ctc_weight 0.5, average_num 20

| decoding mode          | Dev CER | Test CER |
|------------------------|------| ---- |
| ctc greedy search      | 4.86 | 5.46 |
| ctc prefix beam search | 4.86 | 5.46 |
| cif greedy search      | 4.34 | 4.81 |
| cif beam search        | 4.27 | 4.75 |

