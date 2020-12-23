# Performance Record

* Aishell1
* torchaudio.compliance.kalid.fbank
* fbank, dither=0, no cmvn, no speed perturb.
* 80 epoch / 12h,8GPU

| decoding mode            | wav  | kaldi|
|--------------------------|------|------|
| attention decoder        | 5.18 | 5.28 |
| ctc greedy search        | 5.41 | 5.68 |
| ctc prefix beam search   | 5.55 | 5.79 |
| attention rescoring      | 5.55 | 5.78 |


| Exp id |                          | All test | origin(539 utts) | max_pain_wav_n25_0.1(539 utts) | max_pain_wav_n25_0.8(539 utts) | poly_distortion_wav_8_2_2_1.0(539 utts) | double_jag_distortion_wav_0.8(539 utts)  |
|--------|--------------------------|----------|------------------|--------------------------------|--------------------------------|-----------------------------------------|------------------------------------------|
| e1     | torchaudio-100-conformer | 5.18%    | 3.15%            | 66.16%                         | 90.77%                         | 89.61%                                  | 25.78%                                   |
| e2     | d1 + WavDis(20epoch)     | 5.17%    | 3.12 %           | 21.52 %                        | 54.09%                         | 21.52%                                  | 13.31 %                                  |
