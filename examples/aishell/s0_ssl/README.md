# w2v-conformer 

This is a example to use unsupervised pretrained w2v-conformer model to fintune Aishell task.

We pretrain conformer encoders using wav2vec 2.0 pre-training method and we use fbank features as inputs.

The w2v-conformer model uses ISML datasets to pretrain, this is a internal dataset contains 60k hours Chinese.


## pretraining :

We use two model configurations to pretrain the conformer encoder architecture:

Base model contains 12 conformer blocks, model dimension 512, FFN dimension 2048 and 8 attention heads. 
samples are batched together to not exceed 30000 frames per GPU. we train a total of 32 V100 GPUs for 800k steeps.

Middle model contains 24 conformer blocks with model dimension 2048, FFN dimension 512 and 8 attention heads. We add a reconstruction loss to slightly improve performance. To speed up training procedure, we change The time stride of convolutional subsampling blocks to 3, so the length of the input feature becomes one sixth. samples are batched together to not exceed 20000 frames per GPU. we train a total of 32 V100 GPUs for 600k steps.

We are also trying to train the causal model for u2 training and large model with 300m parameters, and this work is ongoing.

pretrained model link:
|   model   | Architecture  | Model |
|-------------------|----|----|
|  Base  |   90Mb  | -
|  Middle |  180M   | - |



## finetuning tips:

*  After pretraining, we can build encoder-decoder based ASR system.The conformer based encoder takes the pretrained model as initialization and the transformer based decoder will be trained from scratch. Just set --enc_init_mods like 'encoder.embed.,encoder.encoders.0.,encoder.encoders.1. ...' to load customized pretrained parameters.

* In aishell task, we carefully adjust the learning rate to 0.0004~0.0005 to get best performence we also find that if too many layers are set for decoder,the migration performance of the pre-training model will be degraded, so we only build a small transformer decoder for joint training. If the downstream task is more than 500 hours, you can increase the learning rate and the parameter amount of the decoder.

* Please note that the final layer of the pretraining model do not provide a good initialization for fine-tuning and would benefit from being re-initialized before fine-tuning. 

# Base model performance

##  Conformer Result

* config: conf/train_conformer_base_100h.yaml
* Training info: lr 0.0004, batch size 16, 4 gpus on A100, acc_grad 1, 250 epochs
* Decoding info: ctc_weight 0.5, average_num 35 

| decoding mode             | CER   |
|---------------------------|-------|
| ctc greedy search    | 3.86  |
| ctc prefix beam search    | 3.86  |
| attention rescoring       | 3.79  |

# Middle model performance

##  Conformer Result

* config: conf/train_conformer_large_100h.yaml
* Training info: lr 0.0005, batch size 16, 4 gpus on A100, acc_grad 1, 250 epochs
* Decoding info: ctc_weight 0.5, average_num 35

| decoding mode             | CER   |
|---------------------------|-------|
| ctc greedy search    |  3.46  |
| ctc prefix beam search    |  3.46 |
| attention rescoring       |  3.37 |


