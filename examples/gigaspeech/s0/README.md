# Performance Record

## Conformer Result

* Feature info: using fbank feature, dither 1.0
* Training info: train_conformer.yaml, kernel size 31, lr 0.002, batch size 48, 8 gpu, acc_grad 4, 15 epochs, use amp
* Decoding info: ctc_weight 0.5, average_num 3
* Git hash: 11aaea7195ffe1014a4bbd16a1fb47ec7dbee2ac
* Model link: http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/gigaspeech/20210115_conformer_exp.tar.gz

| decoding mode                 | test  |
|-------------------------------|-------|
| ctc greedy search             | 13.38 |
| attention rescoring           | 12.80 |
