# Performance Record

## U2++ Conformer Result

* Feature info: using fbank feature, dither, cmvn, oneline speed perturb
* Training info: lr 0.001, dynamic batch with max_frames_in_batch 15000, 4 gpu, acc_grad 1, 130 epochs
* Training weight info: transducer_weight 0.75,  ctc_weight 0.1, reverse_weight 0.30, average_num 30
* Predictor type: lstm

| decoding mode/chunk size  | full  | 16    |
|---------------------------|-------|-------|
| rnnt greedy search        | 6.44  | 7.09  |

