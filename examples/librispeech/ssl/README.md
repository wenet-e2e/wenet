# Performance Record

## Conformer Result  (Base 12layer)

pretrain Conformer 
* pretrain config: conf/pretrain/train_conformer_pretrain_w2v.yaml
* finetune config: conf/finetune/train_conformer_100h.yaml
* beam: 10
* num of gpu: 8
* num of averaged model: 20
* ctc weight (used for attention rescoring): 0.5
* pretrain 90 epochs ,finetune  80 epochs

test set results trained with 100 hours train-clean set

## wav2vec2.0 Results

test clean
| decoding mode   | full |
|--------------------------|------|
| ctc prefix beam search   | 5.77 | 
| attention rescoring      | 5.30 | 

test other
| decoding mode | full | 
|--------------------------|------|
| ctc prefix beam search   | 12.73 | 
| attention rescoring      | 12.14 | 


## data2vec Results

going