# Performance Record

## Conformer Result

* Feature info: dither + specaug + speed perturb
* Training info: lr 0.001, warmup_steps 25000, batch size 16, 1 gpu, acc_grad 4, 240 epochs
* Decoding info: average_num 10

|      decoding mode     |   eval2000 (wer) |
|:----------------------:|:----------------:|
|   ctc_greedy_search    |       32.39%     |
| ctc_prefix_beam_search |       32.39%     |
|         attention      |       31.28%     |
|  attention_rescoring   |       31.36%     |