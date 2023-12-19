# Preliminary
1. Download whisper ckpt from this [link](https://github.com/openai/whisper/blob/main/whisper/__init__.py#L17-L30)

2. We assume you have run stage0~stage3 using `aishell/s0/run.sh` and here we are simply creating a symbolic link
```sh
ln -s ../s0/data .
```

3. Run below command to convert openai-style ckpt to wenet-style ckpt:
```sh
mkdir -p exp
mkdir -p exp/whisper
mkdir -p exp/whisper/large-v3
. ./path.sh && python wenet/whisper/convert_whisper_to_wenet_config_and_ckpt.py \
  --whisper_ckpt downloaded-large-v3.pt \
  --output_dir exp/whisper/large-v3
python local/filter_ckpt.py \
  --filter_list "encoder.embed.conv" \
  --input_ckpt exp/whisper/large-v3/wenet_whisper.pt \
  --output_ckpt exp/whisper/large-v3/wenet_whisper.remove-subsample.pt
```

# Performance Record

## Whisper-largev2 (original) Result

| decoding mode             |  CER  |
|---------------------------|-------|
| attention decoder         | 8.47  |
| ctc greedy search         |  N/A  |
| ctc prefix beam search    |  N/A  |
| attention rescoring       |  N/A  |

## Whisper-largev3 (conv1d2, full-parameter tuning) Result

* Feature info: using log_mel_spectrogram feature, no cmvn, no speed perturb
* Training info: bf16, deepspeed stage1, activation checkpointing, batch dynamic12000, acc_grad 4, 8 * 3090 gpu, 40 epochs (about 14 hours)
* Decoding info: ctc_weight 0.3, average_num 5
* Git hash: TBD

| decoding mode             | CER   |
|---------------------------|-------|
| attention decoder         | 4.06  |
| ctc greedy search         | 8.33  |
| ctc prefix beam search    | 8.34  |
| attention rescoring       | 6.49  |

## Whisper-largev3 (conv2d4, full-parameter tuning) Result

* Feature info: using log_mel_spectrogram feature, no cmvn, no speed perturb
* Training info: bf16, deepspeed stage1, activation checkpointing, batch dynamic12000, acc_grad 4, 8 * 3090 gpu, 40 epochs (about 10 hours)
* Decoding info: ctc_weight 0.3, average_num 5
* Git hash: TBD

| decoding mode             | CER   |
|---------------------------|-------|
| attention decoder         | 3.83  |
| ctc greedy search         | 6.87  |
| ctc prefix beam search    | 6.87  |
| attention rescoring       | 5.33  |

## Whisper-largev3 (conv2d4, full-parameter tuning, HybridTokenizer) Result

* Feature info: using log_mel_spectrogram feature, no cmvn, no speed perturb
* Training info: bf16, deepspeed stage1, activation checkpointing, batch dynamic12000, acc_grad 4, 8 * 3090 gpu, 40 epochs (about 10 hours)
* Decoding info: ctc_weight 0.3, average_num 5
* Git hash: TBD

| decoding mode             | CER   |
|---------------------------|-------|
| attention decoder         | 3.67  |
| ctc greedy search         | 6.18  |
| ctc prefix beam search    | 6.18  |
| attention rescoring       | 4.63  |
