# Performance Record

## Conformer Result

* Feature info: dither + specaug + speed perturb
* Training info: lr 0.002, warmup_steps 5000 batch size 16, 1 gpu, acc_grad 4, 120 epochs
* Decoding info: average_num 20
* trans_type: phn


|     decoding mode      | test (wer) |
| :--------------------: | :---------: |
|   ctc_greedy_search    |   16.70%    |
| ctc_prefix_beam_search |   16.60%    |
|       attention        |   22.37%    |
|  attention_rescoring   |   16.60%    |

## transformer Result

* Feature info: dither + specaug + speed perturb
* Training info: lr 0.002, warmup_steps 5000 batch size 16, 1 gpu, acc_grad 4, 120 epochs
* Decoding info: average_num 20
* trans_type: phn


|     decoding mode      | test (wer) |
| :--------------------: | :---------: |
|   ctc_greedy_search    |   17.78%    |
| ctc_prefix_beam_search |   17.46%    |
|       attention        |   21.77%    |
|  attention_rescoring   |   17.06%    |