# Performance Record

## SenseVoice-small (original) Result

| decoding mode          | full |
|------------------------|------|
| ctc greedy search      | 2.92 % N=104765 C=101787 S=2884 D=94 I=77 |
| ctc prefix beam search | 2.92 % N=104765 C=101786 S=2884 D=95 I=82 |

## SenseVoice-small (full-parameter tuning) Result

* Training info: torch_ddp fp32, ctc_weight: 1.0, acc_grad 1, 8 * 3090 gpu, 49 epochs
* Decoding info: ctc_weight 1.0, average_num 5
* Git hash: TBD

| decoding mode          | full |
|------------------------|------|
| ctc greedy search      | 2.19 % N=104765 C=102543 S=2144 D=78 I=70 |
| ctc prefix beam search | 2.19 % N=104765 C=102543 S=2145 D=77 I=70 |
