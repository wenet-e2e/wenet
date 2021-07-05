# GigaSpeech
A Large, modern and evolving dataset for automatic speech recognition. More details about GigaSpeech can be found:  https://github.com/SpeechColab/GigaSpeech

# Performance Record

## Conformer bidecoder Result

* Feature info: using fbank feature, dither 1.0, no cmvn, 16k
* Training info: conf/train_conformer_bidecoder.yaml, kernel size 31, lr 0.002, batch size 28, 8 gpu, acc_grad 4, 40 epochs
* Decoding info: ctc_weight 0.3, reverse_weight 0.5, average_num 10
* Git hash: 9a0c270f9f976d7e887f777690e6c358a45a1c27
* Model link: http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/gigaspeech/20210705_conformer_bidecoder_exp.tar.gz

## test set gigaspeech scoring

| SPKR      | # Snt |  # Wrd | Corr | Sub | Del | Ins | Err  | S.Err |
|-----------|-------|--------|------|-----|-----|-----|------|-------|
| Sum/Avg   |19928  | 390656 | 91.1 | 6.6 | 2.2 | 2.0 | 10.9 | 64.3  |
| Mean      | 152.1 | 2982.1 | 91.0 | 6.6 | 2.4 | 1.7 | 10.7 | 65.0  |
| S.D.      | 142.2 | 2838.1 |  5.6 | 4.2 | 1.6 | 1.3 |  6.5 | 16.7  |
| Median    | 108.0 | 2000.0 | 92.3 | 5.6 | 2.0 | 1.4 |  8.8 | 65.6  |


## dev set gigaspeech scoring

| SPKR      | # Snt |  # Wrd | Corr | Sub | Del | Ins | Err  | S.Err |
|-----------|-------|--------|------|-----|-----|-----|------|-------|
| Sum/Avg   | 5715  | 127790 | 91.9 | 6.0 | 2.1 | 2.8 | 11.0 | 70.5  |
| Mean      | 204.1 | 4563.9 | 92.6 | 5.5 | 1.9 | 2.1 |  9.5 | 70.4  |
| S.D.      | 269.7 | 4551.6 |  3.6 | 2.9 | 0.9 | 1.7 |  4.9 | 16.5  |
| Median    | 151.5 | 3314.0 | 93.7 | 4.7 | 1.6 | 1.6 |  8.1 | 72.6  |

## Conformer Result

* Feature info: using fbank feature, dither 1.0, no cmvn, 48k
* Training info: conf/train_conformer.yaml, kernel size 31, lr 0.001, batch size 24, 8 gpu, acc_grad 4, 30 epochs
* Decoding info: ctc_weight 0.5, average_num 5
* Git hash: 9a0c270f9f976d7e887f777690e6c358a45a1c27
* Model link: http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/gigaspeech/20210618_conformer_exp.tar.gz

## test set gigaspeech scoring

| SPKR          | # Snt |  # Wrd | Corr | Sub | Del | Ins | Err  | S.Err |
|---------------|-------|--------|------|-----|-----|-----|------|-------|
| Sum/Avg       | 19930 | 390744 | 90.8 | 6.9 | 2.3 | 2.0 | 11.2 | 65.1  |
| Mean          | 152.1 | 2982.8 | 90.6 | 6.9 | 2.5 | 1.7 | 11.1 | 65.7  |
| S.D.          | 142.3 | 2839.0 |  5.8 | 4.3 | 1.7 | 1.2 |  6.7 | 16.6  |
| Median        | 108.0 | 2000.0 | 92.5 | 5.6 | 2.1 | 1.3 |  9.1 | 65.9  |
