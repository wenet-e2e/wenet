# Preliminary
1. Run below command to convert funasr-style  ckpt to wenet-style ckpt:
```sh
output_dir=exp/paraformer/large
mkdir -p ${output_dir}
. ./path.sh && python wenet/paraformer/convert_paraformer_to_wenet_config_and_ckpt.py \
  --output_dir ${output_dir}
# init ctc and embed(used in sampler)
python local/modify_ckpt.py \
  --add_list "{\"ctc.ctc_lo.weight\": \"embed.weight\"}" \
  --input_ckpt exp/paraformer/large/wenet_paraformer.pt \
  --output_ckpt exp/paraformer/large/wenet_paraformer.init-ctc.init-embed.pt
```

# Performance Record

## Paraformer (original) Result

| decoding mode             |  full | 16  |
|---------------------------|-------|-----|
| paraformer greedy search  | 1.95  | N/A |

## Paraformer (full-parameter tuning) Result

* Training info: torch_ddp fp32, batch size 28, ctc_weight: 0.3, acc_grad 1, 8 * 3090 gpu, 60 epochs (about 8h)
* Decoding info: ctc_weight 0.3, average_num 5
* Git hash: TBD

| decoding mode             | full  | 16  |
|---------------------------|-------|-----|
| ctc greedy search         | 3.45 % N=104765 C=101244 S=3406 D=115 I=91  | N/A |
| ctc prefix beam search    | 3.44 % N=104765 C=101247 S=3407 D=111 I=83  | N/A |
| paraformer greedy search  | 2.19 % N=104765 C=102643 S=1959 D=163 I=172 | N/A |

## Paraformer-dynamic training (full-parameter tuning) Result

* Training info: torch_ddp fp32, batch size 28, ctc_weight: 0.3, acc_grad 1, 8 * 3090 gpu, 60 epochs (about 8h)
* Decoding info: ctc_weight 0.3, average_num 5
* Git hash: TBD

| decoding mode             | full   | 16   |
|---------------------------|--------|------|
| ctc greedy search         | 3.46 % N=104765 C=101235 S=3409 D=121 I=98   | 4.18 % N=104765 C=100495 S=4149 D=121 I=107 |
| ctc prefix beam search    | 3.45 % N=104765 C=101239 S=3413 D=113 I=91   | 4.17 % N=104765 C=100500 S=4150 D=115 I=103 |
| paraformer greedy search  | 2.15 % N=104765 C=102640 S=1977 D=148 I=132  | 2.40 % N=104765 C=102409 S=2220 D=136 I=161 |
