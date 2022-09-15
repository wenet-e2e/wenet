# Develop Record
```
squeezeformer
├── attention.py                        # reltive multi-head attention module  
├── conv2d.py                           # self defined conv2d valid padding module
├── convolution.py                      # convolution module in squeezeformer block
├── encoder_layer.py                    # squeezeformer encoder layer
├── encoder.py                          # squeezeformer encoder class 
├── positionwise_feed_forward.py        # feed forward layer 
├── subsampling.py                      # sub-sampling layer, time reduction layer
└── utils.py                            # residual connection module
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
* encoder flops(30s): 2,797,274,624, params: 34,761,608


### Squeezeformer Result (SM12, FFN:1024)
* encoder flops(30s): 21,158,877,440, params: 22,219,912
* Feature info: using fbank feature, cmvn, dither, online speed perturb
* Training info: train_squeezeformer.yaml, kernel size 31, lr 0.001, batch size 12, 8 gpu, acc_grad 4, 120 epochs, dither 0.1
* Decoding info: ctc_weight 0.3, reverse weight 0.5, average_num 30

| decoding mode                    | dev clean | dev other | test clean | test other |
|----------------------------------|-----------|-----------|------------|------------|
| ctc greedy search                | 3.49      | 9.24      | 3.51       | 9.28       |
| ctc prefix beam search           | 3.44      | 9.23      | 3.51       | 9.25       |
| attention decoder                | 8.74      | 3.59      | 3.75       | 8.70       |
| attention rescoring              | 2.97      | 8.48      | 3.07       | 8.44       |