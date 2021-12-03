# Performance Record

## Conformer Result

* Feature info: dither + specaug + speed perturb
* Training info: lr 0.002, warmup_steps 20000 batch size 16, 1 gpu, acc_grad 4, 120 epochs
* Decoding info: average_num 20

|      decoding mode     |   dev93 (cer) |  dev93 (wer)  |
|:----------------------:|:-------------:|:-------------:|
|   ctc_greedy_search    |     5.25%     |    13.16%     |
| ctc_prefix_beam_search |     5.17%     |    13.10%     |
|  attention_rescoring   |     5.11%     |    12.17%     |