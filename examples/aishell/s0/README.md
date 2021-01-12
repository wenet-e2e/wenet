# Performance Record

* Aishell1
* use torchaudio.compliance.kalid.fbank
* fbank, dither=0, no cmvn, no speed perturb.
* 80 epoch / 12h,8GPU

| decoding mode            | wav  | kaldi|
|--------------------------|------|------|
| attention decoder        | 5.18 | 5.28 |
| ctc greedy search        | 5.41 | 5.68 |
| ctc prefix beam search   | 5.55 | 5.79 |
| attention rescoring      | 5.55 | 5.78 |

