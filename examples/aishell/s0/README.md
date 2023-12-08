# Performance Record

## Conformer Result

* Feature info: using fbank feature, dither, cmvn, online speed perturb
* Training info: lr 0.002, batch size 18, 4 gpu, acc_grad 4, 240 epochs, dither 0.1
* Decoding info: ctc_weight 0.5, average_num 20
* Git hash: 919f07c4887ac500168ba84b39b535fd8e58918a

| decoding mode             | CER   |
|---------------------------|-------|
| attention decoder         | 5.18  |
| ctc greedy search         | 4.94  |
| ctc prefix beam search    | 4.94  |
| attention rescoring       | 4.61  |
| LM + attention rescoring  | 4.36  |

## U2++ Conformer Result

* Feature info: using fbank feature, dither=1.0, cmvn, oneline speed perturb
* Training info: lr 0.001, batch size 16, 8 gpu, acc_grad 1, 360 epochs
* Decoding info: ctc_weight 0.3, reverse_weight 0.5  average_num 30, lm_scale 0.7, decoder_scale 0.1, r_decoder_scale 0.7
* Git hash: 5a1342312668e7a5abb83aed1e53256819cebf95

| decoding mode/chunk size  | full  | 16    |
|---------------------------|-------|-------|
| ctc greedy search         | 5.19  | 5.81  |
| ctc prefix beam search    | 5.17  | 5.81  |
| attention rescoring       | 4.63  | 5.05  |
| LM + attention rescoring  | 4.40  | 4.75  |
| HLG(k2 LM)                | 4.81  | 5.27  |
| HLG(k2 LM)  + attention rescoring | 4.32  | 4.70  |
| HLG(k2 LM)  + attention rescoring + LFMMI | 4.11  | 4.47  |

## U2++ lite Conformer Result (uio shard)

* Feature info: using fbank feature, dither=1.0, cmvn, oneline speed perturb
* Training info: lr 0.001, batch size 16, 8 gpu, acc_grad 1, load a well trained model and continue training 80 epochs with u2++ lite config
* Decoding info: ctc_weight 0.3, reverse_weight 0.5  average_num 30
* Git hash: 73185808fa1463b0163a922dc722513b7baabe9e

| decoding mode/chunk size  | full  | 16    |
|---------------------------|-------|-------|
| ctc greedy search         | 5.21  | 5.91  |
| ctc prefix beam search    | 5.20  | 5.91  |
| attention rescoring       | 4.67  | 5.10  |

## Unified Conformer Result

* Feature info: using fbank feature, dither=0, cmvn, oneline speed perturb
* Training info: lr 0.001, batch size 16, 8 gpu, acc_grad 1, 180 epochs, dither 0.0
* Decoding info: ctc_weight 0.5, average_num 20
* Git hash: 919f07c4887ac500168ba84b39b535fd8e58918a

| decoding mode/chunk size  | full  | 16    | 8     | 4     |
|---------------------------|-------|-------|-------|-------|
| attention decoder         | 5.40  | 5.60  | 5.74  | 5.86  |
| ctc greedy search         | 5.56  | 6.29  | 6.68  | 7.10  |
| ctc prefix beam search    | 5.57  | 6.30  | 6.67  | 7.10  |
| attention rescoring       | 5.05  | 5.45  | 5.69  | 5.91  |
| LM + attention rescoring  | 4.73  | 5.08  | 5.22  | 5.38  |

## U2++ Transformer Result

* Feature info: using fbank feature, dither, cmvn, online speed perturb.
* Training info: lr 0.001, batch size 26, 8 gpu, acc_grad 1, 360 epochs, dither 0.1
* Decoding info: ctc_weight 0.2, reverse_weight 0.5, average_num 30
* Git hash: 65270043fc8c2476d1ab95e7c39f730017a670e0

| decoding mode/chunk size  | full  | 16    |
|---------------------------|-------|-------|
| ctc greedy search         | 6.05  | 6.92  |
| ctc prefix beam search    | 6.05  | 6.90  |
| attention rescoring       | 5.11  | 5.63  |
| LM + attention rescoring  | 4.82  | 5.24  |

## Transformer Result

* Feature info: using fbank feature, dither, with cmvn, online speed perturb.
* Training info: lr 0.002, batch size 26, 4 gpu, acc_grad 4, 240 epochs, dither 0.1
* Decoding info: ctc_weight 0.5, average_num 20
* Git hash: 919f07c4887ac500168ba84b39b535fd8e58918a

| decoding mode             | CER   |
|---------------------------|-------|
| attention decoder         | 5.69  |
| ctc greedy search         | 5.92  |
| ctc prefix beam search    | 5.91  |
| attention rescoring       | 5.30  |
| LM + attention rescoring  | 5.04  |

## Unified Transformer Result

* Feature info: using fbank feature, dither=0, with cmvn, online speed perturb.
* Training info: lr 0.002, batch size 16, 4 gpu, acc_grad 1, 240 epochs, dither 0.1
* Decoding info: ctc_weight 0.5, average_num 20
* Git hash: 919f07c4887ac500168ba84b39b535fd8e58918a

| decoding mode/chunk size  | full  | 16    | 8     | 4     |
|---------------------------|-------|-------|-------|-------|
| attention decoder         | 6.04  | 6.35  | 6.45  | 6.70  |
| ctc greedy search         | 6.28  | 6.99  | 7.39  | 7.89  |
| ctc prefix beam search    | 6.28  | 6.98  | 7.40  | 7.89  |
| attention rescoring       | 5.52  | 6.05  | 6.28  | 6.62  |
| LM + attention rescoring  | 5.11  | 5.59  | 5.86  | 6.17  |

## AMP Training Transformer Result

* Feature info: using fbank feature, dither, cmvn, online speed perturb
* Training info: lr 0.002, batch size, 4 gpus, acc_grad 4, 240 epochs, dither 0.1, warm up steps 25000
* Decoding info: ctc_weight 0.5, average_num 20
* Git hash: 1bb4e5a269c535340fae5b0739482fa47733d2c1

| decoding mode          | CER  |
|------------------------|------|
| attention decoder      | 5.73 |
| ctc greedy search      | 5.92 |
| ctc prefix beam search | 5.92 |
| attention rescoring    | 5.31 |


## Muilti-machines Training Conformer Result

* Feature info: using fbank feature, dither, cmvn, online speed perturb
* Training info: lr 0.004, batch size 16, 2 machines, 8\*2=16 gpus, acc_grad 4, 240 epochs, dither 0.1, warm up steps 10000
* Decoding info: ctc_weight 0.5, average_num 20
* Git hash: f6b1409023440da1998d31abbcc3826dd40aaf35

| decoding mode          | CER  |
|------------------------|------|
| attention decoder      | 4.90 |
| ctc greedy search      | 5.07 |
| ctc prefix beam search | 5.06 |
| attention rescoring    | 4.65 |


## Conformer with/without Position Encoding Result

* Feature info: using fbank feature, dither, cmvn, online speed perturb
* Training info: lr 0.002, batch size 16, 8 gpu, acc_grad 4, 240 epochs, dither 0.1
* Decoding info: ctc_weight 0.5, average_num 20

| decoding mode          | with PE | without PE |
|------------------------|---------|------------|
| attention decoder      | 5.18    | 5.73       |
| ctc greedy search      | 4.94    | 4.97       |
| ctc prefix beam search | 4.94    | 4.97       |
| attention rescoring    | 4.61    | 4.69       |


## Efficient Conformer v1 Result

* Feature info:
    * using fbank feature, cmvn, speed perturb, dither
* Training info:
    * train_u2++_efficonformer_v1.yaml
    * 8 gpu, batch size 16, acc_grad 1, 200 epochs
    * lr 0.001, warmup_steps 25000
* Model info:
    * Model Params: 48,488,347
    * Downsample rate: 1/4 (conv2d) * 1/2 (efficonformer block)
    * encoder_dim 256, output_size 256, head 8, linear_units 2048
    * num_blocks 12, cnn_module_kernel 15, group_size 3
* Decoding info:
    * ctc_weight 0.5, reverse_weight 0.3, average_num 20
* Model Download: [wenet_efficient_conformer_aishell_v1](https://huggingface.co/58AILab/wenet_efficient_conformer_aishell_v1)

| decoding mode          | full | 18   | 16   |
|------------------------|------|------|------|
| attention decoder      | 4.99 | 5.13 | 5.16 |
| ctc prefix beam search | 4.98 | 5.23 | 5.23 |
| attention rescoring    | 4.64 | 4.86 | 4.85 |


## Efficient Conformer v2 Result

* Feature info:
    * using fbank feature, cmvn, speed perturb, dither
* Training info:
    * train_u2++_efficonformer_v2.yaml
    * 8 gpu, batch size 16, acc_grad 1, 200 epochs
    * lr 0.001, warmup_steps 25000
* Model info:
    * Model Params: 49,354,651
    * Downsample rate: 1/2 (conv2d2) * 1/4 (efficonformer block)
    * encoder_dim 256, output_size 256, head 8, linear_units 2048
    * num_blocks 12, cnn_module_kernel 15, group_size 3
* Decoding info:
    * ctc_weight 0.5, reverse_weight 0.3, average_num 20
* Model Download: [wenet_efficient_conformer_aishell_v2](https://huggingface.co/58AILab/wenet_efficient_conformer_aishell_v2)

| decoding mode          | full | 18   | 16   |
|------------------------|------|------|------|
| attention decoder      | 4.87 | 5.03 | 5.07 |
| ctc prefix beam search | 4.97 | 5.18 | 5.20 |
| attention rescoring    | 4.56 | 4.75 | 4.77 |


## U2++ Branchformer Result

* Feature info: using fbank feature, dither=1.0, cmvn, oneline speed perturb
* * Model info:
    * Model Params: 48,384,667
    * Num Encoder Layer: 24
    * CNN Kernel Size: 63
    * Merge Method: concat
* Training info: lr 0.001, weight_decay: 0.000001, batch size 16, 3 gpu, acc_grad 1, 360 epochs
* Decoding info: ctc_weight 0.3, reverse_weight 0.5  average_num 30, lm_scale 0.7, decoder_scale 0.1, r_decoder_scale 0.7
* Git hash: 5a1342312668e7a5abb83aed1e53256819cebf95

| decoding mode             | CER   |
|---------------------------|-------|
| ctc greedy search         | 5.28  |
| ctc prefix beam search    | 5.28  |
| attention decoder         | 5.12  |
| attention rescoring       | 4.81  |
| LM + attention rescoring  | 4.46  |

## E-Branchformer Result

* Feature info: using fbank feature, dither=1.0, cmvn, online speed perturb
* * Model info:
    * Model Params: 47,570,132
    * Num Encoder Layer: 17
    * CNN Kernel Size: 31
* Training info: lr 0.001, weight_decay: 0.000001, batch size 16, 4 gpu, acc_grad 1, 240 epochs
* Decoding info: ctc_weight 0.3, average_num 30
* Git hash: 89962d1dcae18dd3a281782a40e74dd2721ae8fe

| decoding mode          | CER  |
| ---------------------- | ---- |
| attention decoder      | 4.73 |
| ctc greedy search      | 4.77 |
| ctc prefix beam search | 4.77 |
| attention rescoring    | 4.39 |
| LM + attention rescoring  | 4.22  |
