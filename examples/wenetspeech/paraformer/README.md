# Preliminary
1. Run below command to convert funasr-style  ckpt to wenet-style ckpt:
```sh
output_dir=exp/paraformer/large
mkdir -p ${output_dir}
. ./path.sh && python wenet/paraformer/convert_paraformer_to_wenet_config_and_ckpt.py \
  --output_dir ${output_dir}
```

# Performance Record

## Paraformer (original) Result

| decoding mode             |  CER(DEV)  | CER(TEST\_NET) | CER(TEST\_MEETING) | CER(AISHELL4-TEST) |
|---------------------------|-------|-------|-------|-------|
| ctc greedy search         |  N/A  |  N/A  |  N/A  |  N/A  |
| ctc prefix beam search    |  N/A  |  N/A  |  N/A  |  N/A  |
| paraformer greedy search  | 3.38 % N=328207 C=319078 S=8045 D=1084 I=1959  | 6.74 % N=414285 C=388817 S=20119 D=5349 I=2444 | 6.95 % N=220358 C=206461 S=9090 D=4807 I=1422 | 18.37 % N=180315 C=156112 S=17714 D=6489 I=8923 |

## Paraformer-dynamic training (full-parameter tuning) Result

* Training info: lr 0.0005, batch size dynamic36000, gradient checkpointing, torch_ddp, acc_grad 4, 8 * 3090 gpus, 20 epochs, about 4 days
* Decoding info: ctc_weight 0.5, average_num 3, blank penalty 2.5 for dev and 0.0 for others
* Git hash: TBD

| decoding mode             |  CER(DEV)  | CER(TEST\_NET) | CER(TEST\_MEETING) | CER(AISHELL4-TEST) |
|---------------------------|-------|-------|-------|-------|
| ctc greedy search         | 7.44 % N=328207 C=307725 S=13700 D=6782 I=3925 | 10.53 % N=414285 C=372693 S=28860 D=12732 I=2013 | 9.60 % N=220358 C=200727 S=12091 D=7540 I=1527 | 20.24 % N=180315 C=149290 S=18264 D=12761 I=5467  |
| ctc prefix beam search    | 7.18 % N=328207 C=308288 S=13896 D=6023 I=3636 | 10.34 % N=414285 C=373390 S=29103 D=11792 I=1929 | 9.26 % N=220358 C=201542 S=12250 D=6566 I=1587 | 19.83 % N=180315 C=150352 S=18592 D=11371 I=5796  |
| paraformer greedy search  | 6.63 % N=328207 C=308518 S=8688 D=11001 I=2077 | 7.71 % N=414285 C=384845 S=20587 D=8853 I=2483 | 7.34 % N=220358 C=205924 S=9283 D=5151 I=1751 | 18.04 % N=180315 C=155176 S=17420 D=7719 I=7386 |


### NOTE

This is our first attempt at fine-tuning the paraformer-large to enable stream inference through a wenet-like chunk method.
Although the non-streaming results deteriorated after fine-tuning compared to before, we believe there is still significant room for improvement for paraformer-large when fine-tuned within wenet, considering this is a very initial result.

Additionally, on the same training set (wenetspeech+aishell4), we trained a conformer-large model from scratch (see experimental results in examples/aishell/s0). Comparing it with the fine-tuned results of paraformer-large, we found that the CTC results of paraformer-large consistently outperformed those of conformer-large, and the NAR results of paraformer-large were always better than the rescore of conformer-large. This is mainly due to paraformer-large having been pre-trained on 6wh industrial data, giving the model a better initialization.
