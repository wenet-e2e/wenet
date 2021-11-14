# Performance Record

## Conformer Result

* Feature info: using fbank feature, cmvn, without speed perturb (not supported segments yet)
* Training info: lr 0.001, max_frames_in_batch 15000, 8 gpu, acc_grad 4, 100 epochs
* Decoding info: ctc_weight 0.5, average_num 30


| decoding mode       | Test WER |
|---------------------|----------|
| attention rescoring |  32.58%  |
