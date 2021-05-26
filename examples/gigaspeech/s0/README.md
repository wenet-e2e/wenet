# GigaSpeech
A Large, modern and evolving dataset for automatic speech recognition. More details about GigaSpeech can be found:  https://github.com/SpeechColab/GigaSpeech

# Performance Record

## Conformer Result

* Feature info: using fbank feature, dither 1.0
* Training info: train_conformer.yaml, kernel size 31, lr 0.002, batch size 48, 8 gpu, acc_grad 4, 15 epochs, use amp
* Decoding info: ctc_weight 0.5, average_num 3
* Git hash: 11aaea7195ffe1014a4bbd16a1fb47ec7dbee2ac
* Model link: http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/gigaspeech/20210520_conformer_exp.tar.gz

# gigaspeech scoring

| SPKR          | # Snt |  # Wrd | Corr | Sub | Del | Ins | Err  | S.Err |
|---------------|-------|--------|------|-----|-----|-----|------|-------|
| Sum/Avg       | 19930 | 390744 | 90.5 | 7.2 | 2.4 | 2.0 | 11.6 | 66.1  |
|     Mean      |152.1  | 2982.8 | 90.2 | 7.3 | 2.5 | 1.7 | 11.5 | 66.6  |
|     S.D.      |142.3  | 2839.0 |  6.2 | 4.7 | 1.8 | 1.2 | 7.1  | 16.8  |
|    Median     |108.0  | 2000.0 | 91.8 | 5.8 | 2.2 | 1.4 | 9.5  | 66.7  |

# the raw scoring version wer

| decoding mode                 | test  |
|-------------------------------|-------|
| ctc greedy search             | 13.38 |
| attention rescoring           | 12.80 |
