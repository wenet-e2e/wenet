# Performance Record

## Conformer Result

* Feature info: using fbank feature, dither=0, no cmvn, no speed perturb.
* Training info: lr 0.001, batch size 16, 8 gpu, acc_grad 1, 80 epochs, dither 0.0
* Git hash: 9da70d74c444a64644f060a29766e0b7a1327719
* Model link: http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/aishell/20210115_conformer_exp.tar.gz

| decoding mode          | CER  |
|------------------------|------|
| attention decoder      | 5.95 |
| ctc greedy search      | 5.89 |
| ctc prefix beam search | 5.89 |
| attention rescoring    | 5.28 |


