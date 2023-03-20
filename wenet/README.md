# Module Introduction

Here is a brief introduction of each module(directory).

* `bin`: training and recognition binaries
* `dataset`: IO design
* `utils`: common utils
* `transformer`: the core of `WeNet`, in which the standard transformer/conformer is implemented. It contains the common blocks(backbone) of speech transformers.
  * transformer/attention.py: Standard multi head attention
  * transformer/embedding.py: Standard position encoding
  * transformer/positionwise_feed_forward.py: Standard feed forward in transformer
  * transformer/convolution.py: ConvolutionModule in Conformer model
  * transformer/subsampling.py: Subsampling implementation for speech task
* `transducer`: transducer implementation
* `squeezeformer`: squeezeformer implementation, please refer [paper](https://arxiv.org/pdf/2206.00888.pdf)
* `efficient_conformer`: efficient conformer implementation, please refer [paper](https://arxiv.org/pdf/2109.01163.pdf)
* `cif`: Continuous Integrate-and-Fire implemented, please refer [paper](https://arxiv.org/pdf/1905.11235.pdf)

`transducer`, `squeezeformer`, `efficient_conformer`, and `cif` are all based on `transformer`,
they resue a lot of the common blocks of `tranformer`.

**If you want to contribute your own x-former, please reuse the current code as much as possible**.


