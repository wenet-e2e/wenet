# Performance Record
# Should be installed ffmpeg , pandas !!!
## Conformer Result

* Feature info: dither + specaug + speed perturb
* Training info: lr 0.0005, warmup_steps 20000 batch size 8, 3 gpu, 30 epochs
* Decoding info: average_num 20



|     decoding mode      | test (wer) |
| :--------------------: | :---------: |
|   ctc_greedy_search    |   16.12%    |
| ctc_prefix_beam_search |   16.07%    |
|       attention        |   13.56%    |
|  attention_rescoring   |   14.01%    |