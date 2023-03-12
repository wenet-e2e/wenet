# Performance Record

## Conformer CIF DecoderSAN + PredictV1 Result

* Feature info: using fbank feature, dither, cmvn, online speed perturb
* Training info: lr 0.002, batch size 16, 4 gpu, acc_grad 4, 240 epochs, dither 0.1
* Decoding info: ctc_weight 0.5, average_num 20

| decoding mode          | Dev CER | Test CER |
|------------------------| --- | ---- |
| ctc greedy search      | 4.65 | 5.24 |
| ctc prefix beam search | 4.65 | 5.24 |
| cif greedy search      | 4.41 | 4.92 |
| cif beam search        | 4.35 | 4.86 |

## Conformer CIF DecoderSAN + PredictV2 Result

* Feature info: using fbank feature, dither, cmvn, online speed perturb
* Training info: lr 0.002, batch size 16, 4 gpu, acc_grad 4, 240 epochs, dither 0.1
* Decoding info: ctc_weight 0.5, average_num 20

| decoding mode          | Dev CER | Test CER |
|------------------------| ---- | ---- |
| ctc greedy search      | 5.35 | 5.98 |
| ctc prefix beam search | 5.35 | 5.98 |
| cif greedy search      | 4.77 | 5.32 |
| cif beam search        | 4.71 | 5.25 |

## Conformer CIF DecoderSANM + PredictV1 Result

* Feature info: using fbank feature, dither, cmvn, online speed perturb
* Training info: lr 0.002, batch size 16, 4 gpu, acc_grad 4, 240 epochs, dither 0.1
* Decoding info: ctc_weight 0.5, average_num 20

| decoding mode          | Dev CER | Test CER |
|------------------------|------| ---- |
| ctc greedy search      | 4.86 | 5.46 |
| ctc prefix beam search | 4.86 | 5.46 |
| cif greedy search      | 4.34 | 4.81 |
| cif beam search        | 4.27 | 4.75 |

## Conformer CIF DecoderSANM + PredictV2 Result

* Feature info: using fbank feature, dither, cmvn, online speed perturb
* Training info: lr 0.002, batch size 16, 4 gpu, acc_grad 4, 240 epochs, dither 0.1
* Decoding info: ctc_weight 0.5, average_num 20

| decoding mode          | Dev CER | Test CER |
|------------------------| ---- | ---- |
| ctc greedy search      | 5.60 | 6.15 |
| ctc prefix beam search | 5.60 | 6.14 |
| cif greedy search      | 4.85 | 5.29 |
| cif beam search        | 4.77 | 5.16 |
