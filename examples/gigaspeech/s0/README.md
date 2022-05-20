# GigaSpeech
A Large, modern and evolving dataset for automatic speech recognition. More details about GigaSpeech can be found:  https://github.com/SpeechColab/GigaSpeech

# Performance Record

## Conformer bidecoder Result

* Feature info: using fbank feature, dither 1.0, cmvn, 16k
* Training info: conf/train_conformer_bidecoder.yaml, subsample 4, kernel size 31, lr 0.001, batch size 24, 8 gpu, acc_grad 4, 40 epochs
* Decoding info: ctc_weight 0.3, reverse_weight 0.5, average_num 10
* Git hash: 9a0c270f9f976d7e887f777690e6c358a45a1c27

### test set gigaspeech scoring

| SPKR      | # Snt |  # Wrd | Corr | Sub | Del | Ins | Err  | S.Err |
|-----------|-------|--------|------|-----|-----|-----|------|-------|
| Sum/Avg   | 19928 | 390656 | 91.4 | 6.4 | 2.2 | 2.0 | 10.6 | 63.1  |
|  Mean     | 152.1 | 2982.1 | 91.4 | 6.3 | 2.3 | 1.7 | 10.3 | 63.7  |
|  S.D.     | 142.2 | 2838.1 |  5.5 | 4.1 | 1.6 | 1.3 |  6.4 | 16.9  |
| Median    | 108.0 | 2000.0 | 93.0 | 5.1 | 2.0 | 1.3 |  8.4 | 64.6  |

### dev set gigaspeech scoring

| SPKR      | # Snt |  # Wrd | Corr | Sub | Del | Ins | Err  | S.Err |
|-----------|-------|--------|------|-----|-----|-----|------|-------|
| Sum/Avg   | 5715  | 127790 | 92.1 | 5.8 | 2.1 | 2.8 | 10.7 |  69.9 |
|  Mean     | 204.1 | 4563.9 | 92.9 | 5.2 | 1.9 | 2.0 |  9.1 |  69.4 |
|  S.D.     | 269.7 | 4551.6 |  3.4 | 2.7 | 0.9 | 1.7 |  4.6 |  15.9 |
| Median    | 151.5 | 3314.0 | 93.8 | 4.4 | 1.6 | 1.7 |  7.9 |  71.6 |

## Conformer U2++ Result

* Feature info: using fbank feature, dither 1.0, cmvn, 16k
* Training info: conf/train_u2++_conformer.yaml, subsample 6, kernel size 31, lr 0.001, batch size 28, 8 gpu, acc_grad 1, 50 epochs
* Decoding info: ctc_weight 0.3, reverse_weight 0.5, average_num 10
* Git hash: 9a0c270f9f976d7e887f777690e6c358a45a1c27

### test set gigaspeech scoring, full chunk (non-streaming)

| SPKR      | # Snt |  # Wrd | Corr | Sub | Del | Ins | Err  | S.Err |
|-----------|-------|--------|------|-----|-----|-----|------|-------|
| Sum/Avg   | 19928 | 390656 | 90.7 | 6.8 | 2.6 | 2.0 | 11.3 |  66.9 |
|  Mean     | 152.1 | 2982.1 | 90.6 | 6.8 | 2.7 | 1.6 | 11.1 |  67.1 |
|  S.D.     | 142.2 | 2838.1 |  5.8 | 4.3 | 1.9 | 1.2 |  6.7 |  16.5 |
| Median    | 108.0 | 2000.0 | 92.1 | 5.7 | 2.2 | 1.3 |  9.0 |  68.9 |

### test set gigaspeech scoring, chunk 8 (latency range from 0 to 480ms)

| SPKR      | # Snt |  # Wrd | Corr | Sub | Del | Ins | Err  | S.Err |
|-----------|-------|--------|------|-----|-----|-----|------|-------|
| Sum/Avg   | 19928 | 390656 | 89.6 | 7.5 | 2.9 | 2.0 | 12.5 |  70.1 |
|  Mean     | 152.1 | 2982.1 | 89.3 | 7.6 | 3.1 | 1.7 | 12.4 |  70.6 |
|  S.D.     | 142.2 | 2838.1 |  6.5 | 4.9 | 2.1 | 1.2 |  7.3 |  15.8 |
| Median    | 108.0 | 2000.0 | 91.1 | 6.3 | 2.5 | 1.4 | 10.2 |  72.2 |

## Conformer Result

* Feature info: using fbank feature, dither 1.0, no cmvn, 48k
* Training info: conf/train_conformer.yaml, kernel size 31, lr 0.001, batch size 24, 8 gpu, acc_grad 4, 30 epochs
* Decoding info: ctc_weight 0.5, average_num 5
* Git hash: 9a0c270f9f976d7e887f777690e6c358a45a1c27

### test set gigaspeech scoring

| SPKR          | # Snt |  # Wrd | Corr | Sub | Del | Ins | Err  | S.Err |
|---------------|-------|--------|------|-----|-----|-----|------|-------|
| Sum/Avg       | 19930 | 390744 | 90.8 | 6.9 | 2.3 | 2.0 | 11.2 | 65.1  |
| Mean          | 152.1 | 2982.8 | 90.6 | 6.9 | 2.5 | 1.7 | 11.1 | 65.7  |
| S.D.          | 142.3 | 2839.0 |  5.8 | 4.3 | 1.7 | 1.2 |  6.7 | 16.6  |
| Median        | 108.0 | 2000.0 | 92.5 | 5.6 | 2.1 | 1.3 |  9.1 | 65.9  |
