# GigaSpeech
A Large, modern and evolving dataset for automatic speech recognition. More details about GigaSpeech can be found:  https://github.com/SpeechColab/GigaSpeech

# Performance Record

## Conformer bidecoder Result

* Feature info: using fbank feature, dither 1.0
* Training info: conf/train_conformer_bidecoder.yaml, kernel size 31, lr 0.002, batch size 24, 8 gpu, acc_grad 4, 30 epochs
* Decoding info: ctc_weight 0.3, reverse_weight 0.5, average_num 5
* Git hash: 9a0c270f9f976d7e887f777690e6c358a45a1c27
* Model link: http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/gigaspeech/20210618_conformer_exp.tar.gz

## gigaspeech scoring
| SPKR          | # Snt |  # Wrd | Corr | Sub | Del | Ins | Err  | S.Err |
|---------------|-------|--------|------|-----|-----|-----|------|-------|
| Sum/Avg       |19619  | 390317 | 90.9 | 6.8 | 2.3 | 2.0 | 11.0 | 65.0  |
| Mean          |149.8  | 2979.5 | 90.9 | 6.7 | 2.4 | 1.7 | 10.8 | 65.5  |
| S.D.          |137.6  | 2834.0 |  5.7 | 4.3 | 1.6 | 1.4 |  6.8 | 17.2  |
| Median        |107.0  | 1998.0 | 92.3 | 5.8 | 2.0 | 1.3 |  9.0 | 64.8  |

## Conformer Result

* Feature info: using fbank feature, dither 1.0
* Training info: conf/train_conformer.yaml, kernel size 31, lr 0.001, batch size 24, 8 gpu, acc_grad 4, 30 epochs
* Decoding info: ctc_weight 0.5, average_num 5
* Git hash: 9a0c270f9f976d7e887f777690e6c358a45a1c27
* Model link: http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/gigaspeech/20210618_conformer_exp.tar.gz

## gigaspeech scoring

| SPKR          | # Snt |  # Wrd | Corr | Sub | Del | Ins | Err  | S.Err |
|---------------|-------|--------|------|-----|-----|-----|------|-------|
| Sum/Avg       | 19930 | 390744 | 90.8 | 6.9 | 2.3 | 2.0 | 11.2 | 65.1  |
| Mean          | 152.1 | 2982.8 | 90.6 | 6.9 | 2.5 | 1.7 | 11.1 | 65.7  |
| S.D.          | 142.3 | 2839.0 |  5.8 | 4.3 | 1.7 | 1.2 |  6.7 | 16.6  |
| Median        | 108.0 | 2000.0 | 92.5 | 5.6 | 2.1 | 1.3 |  9.1 | 65.9  |
