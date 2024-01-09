# Performance Record

## Conformer Result

* Feature info: using fbank feature, cmvn, without speed perturb (not supported segments yet)
* Training info: lr 0.001, max_frames_in_batch 15000, 8 gpu, acc_grad 4, 100 epochs
* Decoding info: ctc_weight 0.5, average_num 30


| decoding mode       | Test WER |
|---------------------|----------|
| attention rescoring |  32.58%  |

## U2pp **Conformer Result

* Feature info: using fbank feature, cmvn
* Training info: lr 0.001, batchsize static16, 8 * 3090 gpu, gradient_checkpoint, 100 epochs, about 8 hours
* Decoding info: ctc_weight 0.5, average_num 30


### Aishell4-test

| decoding mode/chunk size  | full  | 16    |
|---------------------------|-------|-------|
| ctc greedy search         | 32.04 % N=180315 C=128615 S=39491 D=12209 I=6072  | 33.81 % N=180315 C=125431 S=42075 D=12809 I=6083  |
| ctc prefix beam search    | 32.00 % N=180315 C=128931 S=39643 D=11741 I=6318  | 33.77 % N=180315 C=125743 S=42266 D=12306 I=6316  |
| attention                 | 33.41 % N=180315 C=129609 S=39233 D=11473 I=9534  | 35.05 % N=180315 C=127206 S=41063 D=12046 I=10084 |
| attention rescoring       | 31.18 % N=180315 C=130334 S=38618 D=11363 I=6236  | 32.90 % N=180315 C=127223 S=41102 D=11990 I=6223  |

### WenetSpeech-test_meeting

| decoding mode/chunk size  | full  | 16    |
|---------------------------|-------|-------|
| ctc greedy search         | 22.15 % N=220358 C=174426 S=38162 D=7770 I=2872  | 24.13 % N=220358 C=170114 S=41554 D=8690 I=2934  |
| ctc prefix beam search    | 22.10 % N=220358 C=174708 S=38193 D=7457 I=3052  | 24.09 % N=220358 C=170391 S=41622 D=8345 I=3126  |
| attention                 | 22.55 % N=220358 C=175320 S=35853 D=9185 I=4653  | 24.15 % N=220358 C=172277 S=38164 D=9917 I=5142  |
| attention rescoring       | 21.35 % N=220358 C=176208 S=36911 D=7239 I=2886  | 23.27 % N=220358 C=172045 S=40211 D=8102 I=2965  |
