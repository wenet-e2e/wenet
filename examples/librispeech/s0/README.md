# Performance Record

## Conformer Result Bidecoder (large)

* Feature info: using fbank feature, cmvn, dither, online speed perturb
* Training info: train_conformer_bidecoder_large.yaml, kernel size 31, lr 0.002, batch size 12, 8 gpu, acc_grad 4, 120 epochs, dither 1.0
* Decoding info: ctc_weight 0.3, reverse weight 0.5, average_num 30
* Git hash: 65270043fc8c2476d1ab95e7c39f730017a670e0
* LM-tgmed: [3-gram.pruned.1e-7.arpa.gz](http://www.openslr.org/resources/11/3-gram.pruned.1e-7.arpa.gz)
* LM-tglarge: [3-gram.arpa.gz](http://www.openslr.org/resources/11/3-gram.arpa.gz)
* LM-fglarge: [4-gram.arpa.gz](http://www.openslr.org/resources/11/4-gram.arpa.gz)

| decoding mode                    | test clean | test other |
|----------------------------------|------------|------------|
| ctc prefix beam search           | 2.96       | 7.14       |
| attention rescoring              | 2.66       | 6.53       |
| LM-tgmed + attention rescoring   | 2.78       | 6.32       |
| LM-tglarge + attention rescoring | 2.68       | 6.10       |
| LM-fglarge + attention rescoring | 2.65       | 5.98       |

## Conformer Result

* Feature info: using fbank feature, cmvn, dither, online speed perturb
* Training info: train_conformer.yaml, kernel size 31, lr 0.004, batch size 12, 8 gpu, acc_grad 4, 120 epochs, dither 0.1
* Decoding info: ctc_weight 0.5, average_num 30
* Git hash: 90d9a559840e765e82119ab72a11a1f7c1a01b78
* LM-fglarge: [4-gram.arpa.gz](http://www.openslr.org/resources/11/4-gram.arpa.gz)

| decoding mode                    | test clean | test other |
|----------------------------------|------------|------------|
| ctc greedy search                | 3.51       | 9.57       |
| ctc prefix beam search           | 3.51       | 9.56       |
| attention decoder                | 3.05       | 8.36       |
| attention rescoring              | 3.18       | 8.72       |
| attention rescoring (beam 50)    | 3.12       | 8.55       |
| LM-fglarge + attention rescoring | 3.09       | 7.40       |

## Conformer U2++ Result

* Feature info: using fbank feature, cmvn, no speed perturb, dither
* Training info: train_u2++_conformer.yaml lr 0.001, batch size 24, 8 gpu, acc_grad 1, 120 epochs, dither 1.0
* Decoding info: ctc_weight 0.3,  reverse weight 0.5, average_num 30
* Git hash: 65270043fc8c2476d1ab95e7c39f730017a670e0

test clean
| decoding mode                  | full | 16   |
|--------------------------------|------|------|
| ctc prefix beam search         | 3.76 | 4.54 |
| attention rescoring            | 3.32 | 3.80 |

test other
| decoding mode                  | full  | 16    |
|--------------------------------|-------|-------|
| ctc prefix beam search         | 9.50  | 11.52 |
| attention rescoring            | 8.67  | 10.38 |

## Conformer U2 Result

* Feature info: using fbank feature, cmvn, speed perturb, dither
* Training info: train_unified_conformer.yaml lr 0.001, batch size 10, 8 gpu, acc_grad 1, 120 epochs, dither 1.0
* Decoding info: ctc_weight 0.5, average_num 30
* Git hash: 90d9a559840e765e82119ab72a11a1f7c1a01b78
* LM-tgmed: [3-gram.pruned.1e-7.arpa.gz](http://www.openslr.org/resources/11/3-gram.pruned.1e-7.arpa.gz)
* LM-tglarge: [3-gram.arpa.gz](http://www.openslr.org/resources/11/3-gram.arpa.gz)
* LM-fglarge: [4-gram.arpa.gz](http://www.openslr.org/resources/11/4-gram.arpa.gz)

test clean
| decoding mode                    | full | 16   |
|----------------------------------|------|------|
| ctc prefix beam search           | 4.26 | 5.00 |
| attention decoder                | 3.05 | 3.44 |
| attention rescoring              | 3.72 | 4.10 |
| attention rescoring (beam 50)    | 3.57 | 3.95 |
| LM-tgmed + attention rescoring   | 3.56 | 4.02 |
| LM-tglarge + attention rescoring | 3.40 | 3.82 |
| LM-fglarge + attention rescoring | 3.38 | 3.74 |

test other
| decoding mode                    | full  | 16    |
|----------------------------------|-------|-------|
| ctc prefix beam search           | 10.87 | 12.87 |
| attention decoder                | 9.07  | 10.44 |
| attention rescoring              | 9.74  | 11.61 |
| attention rescoring (beam 50)    | 9.34  | 11.13 |
| LM-tgmed + attention rescoring   | 8.78  | 10.26 |
| LM-tglarge + attention rescoring | 8.34  | 9.74  |
| LM-fglarge + attention rescoring | 8.17  | 9.44  |
