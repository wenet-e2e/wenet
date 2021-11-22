# conformer based end-to-end model for VKW challenge

## Standard E2E Results

Conformer without speed perpurb and lm
* config: conf/train_train_vkw_bidirect_12conformer_hs2048_output256_att4_conv2d_char.yaml
* beam: 10
* num of gpu: 8
* num of averaged model: 5
* ctc weight (used for attention rescoring): 0.5

dev set results trained only with training set (785 keywords, 1505 hour train set)

| scenario | Precision | Recall   | F1     | ATWV   |
|----------|-----------|----------|--------|--------|
| lgv      | 0.9281    | 0.6420   | 0.7590 | 0.5183 |
| liv      | 0.8886    | 0.6515   | 0.7518 | 0.6050 |
| stv      | 0.9120    | 0.7471   | 0.8213 | 0.6256 |

dev set results trained with training set and finetune set (785 keywords, 1505 hour train set + 15 hour finetune set)

| scenario | Precision | Recall   | F1     | ATWV   |
|----------|-----------|----------|--------|--------|
| lgv      | 0.9478    | 0.7311   | 0.8255 | 0.6352 |
| liv      | 0.9177    | 0.8398   | 0.8770 | 0.7412 |
| stv      | 0.9320    | 0.8207   | 0.8729 | 0.7120 |

test set results trained only with training set (384 keywords, 1505 hour train set)

| scenario | Precision | Recall   | F1     | ATWV   |
|----------|-----------|----------|--------|--------|
| lgv      | 0.6262    | 0.5648   | 0.5939 | 0.5825 |
| liv      | 0.8797    | 0.6282   | 0.7330 | 0.6061 |
| stv      | 0.9102    | 0.7221   | 0.8053 | 0.6682 |

test set results trained with training set and finetune set (384 keywords, 1505 hour train set + 15 hour finetune set)

| scenario | Precision | Recall   | F1     | ATWV   |
|----------|-----------|----------|--------|--------|
| lgv      | 0.6469    | 0.6276   | 0.6371 | 0.6116 |
| liv      | 0.9278    | 0.7560   | 0.8331 | 0.6927 |
| stv      | 0.9434    | 0.8061   | 0.8693 | 0.7275 |
