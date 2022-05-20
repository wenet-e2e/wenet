# w2v-conformer based end-to-end model for Openasr2021 challenge

This is a example to use unsupervised pretrained w2v-conformer model to fintune [OpenASR2021](https://www.nist.gov/itl/iad/mig/openasr-challenge) constrained-plus tasks.

We pretrain conformer encoders using wav2vec 2.0 pre-training method , which we called ch-w2v-conformer. The original pre-training works take raw waveforms
as input. Unlike these works, we use MFCC features as inputs.

The ch-w2v-conformer model uses following datasets to pretrain:

ISML datasets (6 languages,70k hours): internal dataset contains 40k hours Chinese, Cantonese, Tibetan, Inner Mongolian, Inner Kazakh, Uighur.

Babel datasets (17 languages, 2k hours): Assamese, Bengali, Cantonese, Cebuano, Georgian, Haitian, Kazakh, Kurmanji, Lao, Pashto, Swahili, Tagalog, Tamil, Tok, Turkish, Vietnamese, Zulu

After pretraining, we build ASR system based on CTC-Attention structure. In very low resource task, we find that if too many initialization network structures are constructed in the upper layer of pre-training conformer encoder, the migration performance of the pre-training model will be destroyed, so we only build a single-layer transformer decoder for joint training.

pretrained model link: https://huggingface.co/emiyasstar/ch-w2v-conformer


## constrained-plus Task Performance

* Languages: Cantonese,mongolian,kazakh
* config: conf/train_conformer_large_10h.yaml
* Feature info: using mfcc feature, with dither 1.0, without cmvn
* Training info: lr 0.001, batch size 10, 4 gpus on V100, acc_grad 1, 80 epochs
* Decoding info: ctc_weight 0.5, average_num 35

dev set results trained only with 10 hours training set

## w2v-Conformer

|   decoding_method   | Cantonese(CER)  | mongolian(WER) |
|:-------------------:|:----:|:----:|
|  ctc_greedy_search  | 31.46 | 53.64 |
|  ctc_prefix_search |  31.47   | 53.50 |
| attention_rescoring | 31.45 |  52.96 |

## Conformer （train from scratch）


|   decoding_method   |  Cantonese(CER)  | mongolian(WER) |
|:-------------------:|----:|:----:|
|  ctc_greedy_search  | 61.43 | 89.38 |
|  ctc_prefix_search |  61.37   | 89.53|
| attention_rescoring | 60.61 | 89.60|
