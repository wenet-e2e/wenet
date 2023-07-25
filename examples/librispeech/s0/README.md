# Performance Record

## Conformer Result Bidecoder (large)

* Encoder FLOPs(30s): 96,238,430,720, params: 85,709,704
* Feature info: using fbank feature, cmvn, dither, online speed perturb
* Training info: train_conformer_bidecoder_large.yaml, kernel size 31, lr 0.002, batch size 12, 8 gpu, acc_grad 4, 120 epochs, dither 1.0
* Decoding info: ctc_weight 0.3, reverse weight 0.5, average_num 30
* Git hash: 65270043fc8c2476d1ab95e7c39f730017a670e0
* LM-tgmed: [3-gram.pruned.1e-7.arpa.gz](http://www.openslr.org/resources/11/3-gram.pruned.1e-7.arpa.gz)
* LM-tglarge: [3-gram.arpa.gz](http://www.openslr.org/resources/11/3-gram.arpa.gz)
* LM-fglarge: [4-gram.arpa.gz](http://www.openslr.org/resources/11/4-gram.arpa.gz)

| decoding mode                    | test clean | test other |
|----------------------------------|------------|------------|
| ctc prefix beam search           | 2.96       | 7.14       |
| attention rescoring              | 2.66       | 6.53       |
| LM-tgmed + attention rescoring   | 2.78       | 6.32       |
| LM-tglarge + attention rescoring | 2.68       | 6.10       |
| LM-fglarge + attention rescoring | 2.65       | 5.98       |

## SqueezeFormer Result (U2++, FFN:2048)

* Encoder info:
    * SM12, reduce_idx 5, recover_idx 11, conv1d, batch_norm, syncbn
    * encoder_dim 512, output_size 512, head 8, ffn_dim 512*4=2048
    * Encoder FLOPs(30s): 82,283,704,832, params: 85,984,648
* Feature info:
    * using fbank feature, cmvn, dither, online speed perturb, spec_aug
* Training info:
    * train_squeezeformer_bidecoder_large.yaml, kernel size 31
    * batch size 12, 8 gpu, acc_grad 4, 120 epochs, dither 1.0
    * adamw, lr 8e-4, NoamHold, warmup 0.2, hold 0.3, lr_decay 1.0
* Decoding info:
    * ctc_weight 0.3, reverse weight 0.5, average_num 30

| decoding mode                    | dev clean | dev other | test clean | test other |
|----------------------------------|-----------|-----------|------------|------------|
| ctc greedy search                | 2.55      | 6.62      | 2.73       | 6.59       |
| ctc prefix beam search           | 2.53      | 6.60      | 2.72       | 6.52       |
| attention decoder                | 2.93      | 6.56      | 3.31       | 6.47       |
| attention rescoring              | 2.19      | 6.06      | 2.45       | 5.85       |

## Conformer Result

* Encoder FLOPs(30s): 34,085,088,512, params: 34,761,608
* Feature info: using fbank feature, cmvn, dither, online speed perturb
* Training info: train_conformer.yaml, kernel size 31, lr 0.004, batch size 12, 8 gpu, acc_grad 4, 120 epochs, dither 0.1
* Decoding info: ctc_weight 0.5, average_num 30
* Git hash: 90d9a559840e765e82119ab72a11a1f7c1a01b78
* LM-fglarge: [4-gram.arpa.gz](http://www.openslr.org/resources/11/4-gram.arpa.gz)

| decoding mode                    | test clean | test other |
|----------------------------------|------------|------------|
| ctc greedy search                | 3.51       | 9.57       |
| ctc prefix beam search           | 3.51       | 9.56       |
| attention decoder                | 3.05       | 8.36       |
| attention rescoring              | 3.18       | 8.72       |
| attention rescoring (beam 50)    | 3.12       | 8.55       |
| LM-fglarge + attention rescoring | 3.09       | 7.40       |

## Conformer Result (12 layers, FFN:2048)
* Encoder FLOPs(30s): 34,085,088,512, params: 34,761,608
* Feature info: using fbank feature, cmvn, dither, online speed perturb
* Training info: train_squeezeformer.yaml, kernel size 31,
* batch size 12, 8 gpu, acc_grad 4, 120 epochs, dither 0.1
* AdamW, lr 1e-3, NoamHold, warmup 0.2, hold 0.3, lr_decay 1.0
* Decoding info: ctc_weight 0.3, reverse weight 0.5, average_num 30

| decoding mode                    | dev clean | dev other | test clean | test other |
|----------------------------------|-----------|-----------|------------|------------|
| ctc greedy search                | 3.49      | 9.59      | 3.66       | 9.59       |
| ctc prefix beam search           | 3.49      | 9.61      | 3.66       | 9.55       |
| attention decoder                | 3.52      | 9.04      | 3.85       | 8.97       |
| attention rescoring              | 3.10      | 8.91      | 3.29       | 8.81       |

## SqueezeFormer Result (SM12, FFN:1024)
* Encoder info:
    * SM12, reduce_idx 5, recover_idx 11, conv2d, w/o syncbn
    * encoder_dim 256, output_size 256, head 4, ffn_dim 256*4=1024
    * Encoder FLOPs(30s): 21,158,877,440, params: 22,219,912
* Feature info:
    * using fbank feature, cmvn, dither, online speed perturb
* Training info:
    * train_squeezeformer.yaml, kernel size 31,
    * batch size 12, 8 gpu, acc_grad 4, 120 epochs, dither 0.1
    * adamw, lr=1e-3, noamhold, warmup=0.2, hold=0.3, lr_decay=1.0
* Decoding info: ctc_weight 0.3, reverse weight 0.5, average_num 30

| decoding mode                    | dev clean | dev other | test clean | test other |
|----------------------------------|-----------|-----------|------------|------------|
| ctc greedy search                | 3.49      | 9.24      | 3.51       | 9.28       |
| ctc prefix beam search           | 3.44      | 9.23      | 3.51       | 9.25       |
| attention decoder                | 3.59      | 8.74      | 3.75       | 8.70       |
| attention rescoring              | 2.97      | 8.48      | 3.07       | 8.44       |

## SqueezeFormer Result (SM12, FFN:2048)
* Encoder info:
    * SM12, reduce_idx 5, recover_idx 11, conv2d, w/o syncbn
    * encoder_dim 256, output_size 256, head 4, ffn_dim 256*8=2048
    * encoder FLOPs(30s): 28,230,473,984, params: 34,827,400
* Feature info: using fbank feature, cmvn, dither, online speed perturb
* Training info:
    * train_squeezeformer.yaml, kernel size 31
    * batch size 12, 8 gpu, acc_grad 4, 120 epochs, dither 0.1
    * adamw, lr 1e-3, noamhold, warmup 0.2, hold 0.3, lr_decay 1.0
* Decoding info:
    * ctc_weight 0.3, reverse weight 0.5, average_num 30

| decoding mode                    | dev clean | dev other | test clean | test other |
|----------------------------------|-----------|-----------|------------|------------|
| ctc greedy search                | 3.34      | 9.01      | 3.47       | 8.85       |
| ctc prefix beam search           | 3.33      | 9.02      | 3.46       | 8.81       |
| attention decoder                | 3.64      | 8.62      | 3.91       | 8.33       |
| attention rescoring              | 2.89      | 8.34      | 3.10       | 8.03       |

## SqueezeFormer Result (SM12, FFN:1312)
* Encoder info:
    * SM12, reduce_idx 5, recover_idx 11, conv1d, w/o syncbn
    * encoder_dim 328, output_size 256, head 4, ffn_dim 328*4=1312
    * encoder FLOPs(30s): 34,103,960,008, params: 35,678,352
* Feature info:
    * using fbank feature, cmvn, dither, online speed perturb
* Training info:
    * train_squeezeformer.yaml, kernel size 31,
    * batch size 12, 8 gpu, acc_grad 4, 120 epochs, dither 1.0
    * adamw, lr 1e-3, noamhold, warmup 0.2, hold 0.3, lr_decay 1.0
* Decoding info:
    * ctc_weight 0.3, reverse weight 0.5, average_num 30

| decoding mode                    | dev clean | dev other | test clean | test other |
|----------------------------------|-----------|-----------|------------|------------|
| ctc greedy search                | 3.20      | 8.46      | 3.30       | 8.58       |
| ctc prefix beam search           | 3.18      | 8.44      | 3.30       | 8.55       |
| attention decoder                | 3.38      | 8.31      | 3.89       | 8.32       |
| attention rescoring              | 2.81      | 7.86      | 2.96       | 7.91       |

## Conformer U2++ Result

* Feature info: using fbank feature, cmvn, no speed perturb, dither
* Training info: train_u2++_conformer.yaml lr 0.001, batch size 24, 8 gpu, acc_grad 1, 120 epochs, dither 1.0
* Decoding info: ctc_weight 0.3,  reverse weight 0.5, average_num 30
* Git hash: 65270043fc8c2476d1ab95e7c39f730017a670e0

test clean

| decoding mode                  | full | 16   |
|--------------------------------|------|------|
| ctc prefix beam search         | 3.76 | 4.54 |
| attention rescoring            | 3.32 | 3.80 |

test other

| decoding mode                  | full  | 16    |
|--------------------------------|-------|-------|
| ctc prefix beam search         | 9.50  | 11.52 |
| attention rescoring            | 8.67  | 10.38 |

## SqueezeFormer Result (U2++, FFN:2048)

* Encoder info:
    * SM12, reduce_idx 5, recover_idx 11, conv1d, layer_norm
    * do_rel_shift false, warp_for_time, syncbn
    * encoder_dim 256, output_size 256, head 4, ffn_dim 256*8=2048
    * Encoder FLOPs(30s): 28,255,337,984, params: 34,893,704
* Feature info:
    * using fbank feature, cmvn, dither, online speed perturb
* Training info:
    * train_squeezeformer.yaml, kernel size 31
    * batch size 12, 8 gpu, acc_grad 2, 120 epochs, dither 1.0
    * adamw, lr 8e-4, NoamHold, warmup 0.2, hold 0.3, lr_decay 1.0
* Decoding info:
    * ctc_weight 0.3, reverse weight 0.5, average_num 30

test clean

| decoding mode                  | full | 16   |
|--------------------------------|------|------|
| ctc prefix beam search         | 3.45 | 4.34 |
| attention rescoring            | 3.07 | 3.71 |

test other

| decoding mode                  | full  | 16    |
|--------------------------------|-------|-------|
| ctc prefix beam search         | 8.29  | 10.60 |
| attention rescoring            | 7.58  | 9.60  |

## Branchformer U2++ Result

* Feature info: using fbank feature, cmvn, online speed perturb, dither
* Encoder info: layer num 24, cnn_kernel size 63, cgmlp linear units: 2048
* Training info: train_u2++_branchformer.yaml lr 0.001, batch size 16, 8 gpu, acc_grad 1, 120 epochs, dither 1.0
* Decoding info: ctc_weight 0.3,  reverse weight 0.5, average_num 30

test clean

| decoding mode                  | full | 16   |
|--------------------------------|------|------|
| ctc prefix beam search         | 3.78 | 4.60 |
| attention rescoring            | 3.33 | 3.83 |

test other

| decoding mode                  | full  | 16    |
|--------------------------------|-------|-------|
| ctc prefix beam search         | 9.51  | 11.50 |
| attention rescoring            | 8.76  | 10.34 |


## Conformer U2 Result

* Feature info: using fbank feature, cmvn, speed perturb, dither
* Training info: train_unified_conformer.yaml lr 0.001, batch size 10, 8 gpu, acc_grad 1, 120 epochs, dither 1.0
* Decoding info: ctc_weight 0.5, average_num 30
* Git hash: 90d9a559840e765e82119ab72a11a1f7c1a01b78
* LM-tgmed: [3-gram.pruned.1e-7.arpa.gz](http://www.openslr.org/resources/11/3-gram.pruned.1e-7.arpa.gz)
* LM-tglarge: [3-gram.arpa.gz](http://www.openslr.org/resources/11/3-gram.arpa.gz)
* LM-fglarge: [4-gram.arpa.gz](http://www.openslr.org/resources/11/4-gram.arpa.gz)

test clean

| decoding mode                    | full | 16   |
|----------------------------------|------|------|
| ctc prefix beam search           | 4.26 | 5.00 |
| attention decoder                | 3.05 | 3.44 |
| attention rescoring              | 3.72 | 4.10 |
| attention rescoring (beam 50)    | 3.57 | 3.95 |
| LM-tgmed + attention rescoring   | 3.56 | 4.02 |
| LM-tglarge + attention rescoring | 3.40 | 3.82 |
| LM-fglarge + attention rescoring | 3.38 | 3.74 |

test other

| decoding mode                    | full  | 16    |
|----------------------------------|-------|-------|
| ctc prefix beam search           | 10.87 | 12.87 |
| attention decoder                | 9.07  | 10.44 |
| attention rescoring              | 9.74  | 11.61 |
| attention rescoring (beam 50)    | 9.34  | 11.13 |
| LM-tgmed + attention rescoring   | 8.78  | 10.26 |
| LM-tglarge + attention rescoring | 8.34  | 9.74  |
| LM-fglarge + attention rescoring | 8.17  | 9.44  |


## Efficient Conformer V1 Result

* Feature info:
    * using fbank feature, cmvn, speed perturb, dither
* Training info:
    * train_u2++_efficonformer_v1.yaml
    * 8 gpu, batch size 16, acc_grad 1, 120 epochs
    * lr 0.001, warmup_steps 35000
* Model info:
    * Model Params: 49,474,974
    * Downsample rate: 1/4 (conv2d) * 1/2 (efficonformer block)
    * encoder_dim 256, output_size 256, head 8, linear_units 2048
    * num_blocks 12, cnn_module_kernel 15, group_size 3
* Decoding info:
    * ctc_weight 0.5, reverse_weight 0.3, average_num 20
* Model Download: [wenet_efficient_conformer_librispeech_v1](https://huggingface.co/58AILab/wenet_efficient_conformer_librispeech_v1)

test clean

| decoding mode          | full | 18   | 16   |
|------------------------|------|------|------|
| attention decoder      | 3.65 | 3.88 | 3.87 |
| ctc_greedy_search      | 3.46 | 3.79 | 3.77 |
| ctc prefix beam search | 3.44 | 3.75 | 3.74 |
| attention rescoring    | 3.17 | 3.44 | 3.41 |

test other

| decoding mode          | full | 18    | 16    |
|------------------------|------|-------|-------|
| attention decoder      | 8.51 | 9.24  | 9.25  |
| ctc_greedy_search      | 8.94 | 10.04 | 10.06 |
| ctc prefix beam search | 8.91 | 10    | 10.01 |
| attention rescoring    | 8.21 | 9.25  | 9.25  |


## Efficient Conformer V2 Result

* Feature info:
    * using fbank feature, cmvn, speed perturb, dither
* Training info:
    * train_u2++_efficonformer_v2.yaml
    * 8 gpu, batch size 16, acc_grad 1, 120 epochs
    * lr 0.001, warmup_steps 35000
* Model info:
    * Model Params: 50,341,278
    * Downsample rate: 1/2 (conv2d2) * 1/4 (efficonformer block)
    * encoder_dim 256, output_size 256, head 8, linear_units 2048
    * num_blocks 12, cnn_module_kernel 15, group_size 3
* Decoding info:
    * ctc_weight 0.5, reverse_weight 0.3, average_num 20
* Model Download: [wenet_efficient_conformer_librispeech_v2](https://huggingface.co/58AILab/wenet_efficient_conformer_librispeech_v2)

test clean

| decoding mode          | full | 18   | 16   |
|------------------------|------|------|------|
| attention decoder      | 3.49 | 3.71 | 3.72 |
| ctc_greedy_search      | 3.49 | 3.74 | 3.77 |
| ctc prefix beam search | 3.47 | 3.72 | 3.74 |
| attention rescoring    | 3.12 | 3.38 | 3.36 |

test other

| decoding mode          | full | 18   | 16   |
|------------------------|------|------|------|
| attention decoder      | 8.15 | 9.05 | 9.03 |
| ctc_greedy_search      | 8.73 | 9.82 | 9.83 |
| ctc prefix beam search | 8.70 | 9.81 | 9.79 |
| attention rescoring    | 8.05 | 9.08 | 9.10 |
