# Develop Record

```
squeezeformer
├── attention.py                    # reltive multi-head attention module
├── conv2d.py                       # self defined conv2d valid padding module
├── convolution.py                  # convolution module in squeezeformer block
├── encoder_layer.py                # squeezeformer encoder layer
├── encoder.py                      # squeezeformer encoder class
├── positionwise_feed_forward.py    # feed forward layer
├── subsampling.py                  # sub-sampling layer, time reduction layer
└── utils.py                        # residual connection module
```

* Implementation Details
    * Squeezeformer Encoder
        * [x] add pre layer norm before squeezeformer block
        * [x] derive time reduction layer from tensorflow version
        * [x] enable adaptive scale operation
        * [x] enable init weights for deep model training
        * [x] enable training config and results
        * [x] enable dynamic chunk and JIT export
    * Training
        * [x] enable NoamHoldAnnealing schedular

# Performance Record

### Conformer
* Encoder FLOPs(30s): 34,085,088,512, params: 34,761,608
* Feature info: using fbank feature, cmvn, dither, online speed perturb
* Training info: train_conformer.yaml, kernel size 31, lr 0.004,
* batch size 12, 8 gpu, acc_grad 4, 120 epochs, dither 0.1
* Decoding info: ctc_weight 0.5, average_num 30

| decoding mode                    | test clean | test other |
|----------------------------------|------------|------------|
| ctc greedy search                | 3.51       | 9.57       |
| ctc prefix beam search           | 3.51       | 9.56       |
| attention decoder                | 3.05       | 8.36       |
| attention rescoring              | 3.18       | 8.72       |

### Conformer Result (12 layers, FFN:2048)
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

### Squeezeformer Result (SM12, FFN:1024)
* Encoder info:
    * SM12, reduce_idx 5, recover_idx 11, conv2d
    * encoder_dim 256, output_size 256, head 4, ffn_dim 256*8=2048
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

### Squeezeformer Result (SM12, FFN:2048)
* Encoder info:
    * SM12, reduce_idx 5, recover_idx 11, conv2d
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

### Squeezeformer Result (SM12, FFN:1312)
* Encoder info:
    * SM12, reduce_idx 5, recover_idx 11, conv1d
    * encoder_dim 328, output_size 256, head 4, ffn_dim 328*4=1312
    * encoder FLOPs(30s): 34,103,960,008, params: 35,678,352
* Feature info:
    * using fbank feature, cmvn, dither, online speed perturb
* Training info:
    * train_squeezeformer.yaml, kernel size 31,
    * batch size 12, 8 gpu, acc_grad 4, 120 epochs, dither 1.0
    * adamw, lr 5e-4, noamhold, warmup 0.2, hold 0.3, lr_decay 1.0
* Decoding info:
    * ctc_weight 0.3, reverse weight 0.5, average_num 30

| decoding mode                    | dev clean | dev other | test clean | test other |
|----------------------------------|-----------|-----------|------------|------------|
| ctc greedy search                | 3.20      | 8.46      | 3.30       | 8.58       |
| ctc prefix beam search           | 3.18      | 8.44      | 3.30       | 8.55       |
| attention decoder                | 3.38      | 8.31      | 3.89       | 8.32       |
| attention rescoring              | 2.81      | 7.86      | 2.96       | 7.91       |
