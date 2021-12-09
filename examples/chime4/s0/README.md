# Performance Record

## Conformer Result

* Feature info: dither + specaug + speed perturb
* Training info: lr 0.0005, batch size 8, 1 gpu, acc_grad 4, 80 epochs
* Decoding info: average_num 10

|      decoding mode     | dt05_real_1ch | dt05_simu_1ch | et05_real_1ch | et05_simu_1ch |
|:----------------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| ctc_prefix_beam_search |   19.06%      |   21.17%      |   28.39%      |    29.16%     |
|  attention_rescoring   |   17.92%      |   20.22%      |   27.40%      |    28.25%     |
