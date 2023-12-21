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
* `paraformer`: paraformer implementation, please refer [paper](https://arxiv.org/pdf/1905.11235.pdf)
   * `paraformer/cif.py`: Continuous Integrate-and-Fire implemented, please refer [paper](https://arxiv.org/pdf/1905.11235.pdf)
* `branchformer`: branchformer implementation, please refer [paper](https://arxiv.org/abs/2207.02971)
* `whisper`: whisper implementation, please refer [paper](https://arxiv.org/abs/2212.04356)
* `ssl`: Self-supervised speech model implementation. e.g. wav2vec2, bestrq, w2vbert.
* `ctl_model`: Enhancing the Unified Streaming and Non-streaming Model with  with Contrastive Learning implementation [paper](https://arxiv.org/abs/2306.00755)

`transducer`, `squeezeformer`, `efficient_conformer`, `branchformer` and `cif` are all based on `transformer`,
they resue a lot of the common blocks of `tranformer`.

**If you want to contribute your own x-former, please reuse the current code as much as possible**.


