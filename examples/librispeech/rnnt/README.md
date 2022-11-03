# Performance Record

## Conformer Bidecoder Transducer Result

* Feature info: using fbank feature, dither, cmvn, online speed perturb
* Training info: lr 0.001, dynamic batch with max_frames_in_batch 4000, 8 gpu, acc_grad 1, 60 epochs
* Training weight info: transducer_weight 0.75,  ctc_weight 0.1, reverse_weight 0.30, average_num 10
* Predictor type: lstm

| decoding mode         | dev_clean  | dev_other | test_clean | test_other |
|-----------------------|------------|-----------|------------|------------|
| rnnt_greedy_search    | 3.42%      | 8.99%     |    3.56%   |   9.15%    |
| rnnt_beam_search      | 3.35%      | 8.77%     |    3.45%   |   8.78%    |
| rnnt_beam_att_rescore | 3.25%      | 8.66%     |    3.41%   |   8.68%    |

Pretrained model: https://huggingface.co/yuekai/wenet-asr-librispeech-conformer-transducer-mtl/blob/main/exp/conformer_transducer/avg_10.pt

