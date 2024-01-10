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

| decoding mode             |  CER  |
|---------------------------|-------|
| paraformer greedy search  | 1.95  |

## Paraformer (full-parameter tuning) Result

* Training info: batch size 28, ctc_weight: 0.3, acc_grad 4, 8 * v100 gpu, 40 epochs
* Decoding info: ctc_weight 0.3, average_num 5
* Git hash: TBD

| decoding mode             | CER   |
|---------------------------|-------|
| ctc greedy search         | 4.00  |
| ctc prefix beam search    | 4.00  |
| paraformer greedy search  | 2.16  |

## Paraformer-dynamic training (full-parameter tuning) Result

* Training info: batch size 28, ctc_weight: 0.3, acc_grad 4, 8 * v100 gpu, 43 epochs
* Decoding info: ctc_weight 0.3, average_num 5
* Git hash: TBD

| decoding mode             | full   | 16   |
|---------------------------|--------|------|
| ctc greedy search         | 3.93   | 4.94 |
| ctc prefix beam search    | 3.93   | 4.94 |
| paraformer greedy search  | 2.08   | 2.41 |
