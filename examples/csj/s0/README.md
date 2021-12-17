# Performance Record

## Conformer Result Bidecoder (large)


## Conformer Result

* Feature info: using fbank feature, cmvn, dither, online speed perturb
* Training info: train_conformer.yaml, kernel size 15, lr 0.004, batch size 12, 8 gpu, acc_grad 1, 50 epochs, dither 0.0
* Decoding info: ctc_weight 0.5, average_num 10


| decoding mode                    | test1      | test2      | test3      |
|----------------------------------|------------|------------|------------|
| ctc greedy search                | 7.94       | 5.29       | 6.10       |
| ctc prefix beam search           | 7.83+      | 5.28       | 6.08       |
| attention decoder                | 7.83       | 5.63       | 6.37       |
| attention rescoring              | 7.28+      | 4.81       | 5.44       |

note that "+" means we removed two <0.1s wav files in test1 before decoding.




## Conformer U2++ Result


## Conformer U2 Result

