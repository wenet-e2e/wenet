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

* Feature info: using log_mel_spectrogram feature, no cmvn
* Training info: bf16, deepspeed stage1, activation checkpointing, batch dynamic12000, acc_grad 1, 8 * 3090 gpu, 30 epochs (about 16 hours), conf/finetune_whisper_largev3_onlyattn.yaml
* Decoding info: ctc_weight 0.0, average_num 2
* Git hash: TBD

| decoding mode             | CER   |
|---------------------------|-------|
| attention decoder         | 2.57 % N=104765 C=102142 S=2529 D=94 I=74 |
| ctc greedy search         | N/A |
| ctc prefix beam search    | N/A |
| attention rescoring       | N/A |

* Feature info: using log_mel_spectrogram feature, no cmvn, no speed perturb
* Training info: bf16, deepspeed stage1, activation checkpointing, batch dynamic12000, acc_grad 4, 8 * 3090 gpu, 40 epochs (about 14 hours), conf/finetune_whisper_largev3.yaml
* Decoding info: ctc_weight 0.3, average_num 5
* Git hash: TBD

| decoding mode             | CER   |
|---------------------------|-------|
| attention decoder         | 4.06 % N=104765 C=100643 S=4006 D=116 I=128  |
| ctc greedy search         | 8.33 % N=104765 C=96781 S=7776 D=208 I=747   |
| ctc prefix beam search    | 8.34 % N=104765 C=96787 S=7775 D=203 I=760   |
| attention rescoring       | 6.49 % N=104765 C=98199 S=6427 D=139 I=237   |

## Whisper-largev3 (conv2d4, full-parameter tuning) Result

* Feature info: using log_mel_spectrogram feature, no cmvn
* Training info: bf16, deepspeed stage1, activation checkpointing, batch dynamic12000, acc_grad 1, 8 * 3090 gpu, 30 epochs (about 14 hours), conf/finetune_whisper_largev3_conv2d4_onlyattn.yaml
* Decoding info: ctc_weight 0.0, average_num 2
* Git hash: TBD

| decoding mode             | CER   |
|---------------------------|-------|
| attention decoder         | 2.63 % N=104765 C=102088 S=2579 D=98 I=79  |
| ctc greedy search         | N/A |
| ctc prefix beam search    | N/A |
| attention rescoring       | N/A |

* Feature info: using log_mel_spectrogram feature, no cmvn, no speed perturb
* Training info: bf16, deepspeed stage1, activation checkpointing, batch dynamic12000, acc_grad 4, 8 * 3090 gpu, 40 epochs (about 10 hours), conf/finetune_whisper_largev3_conv2d4.yaml
* Decoding info: ctc_weight 0.3, average_num 5
* Git hash: TBD

| decoding mode             | CER   |
|---------------------------|-------|
| attention decoder         | 3.83 % N=104765 C=100866 S=3784 D=115 I=109 |
| ctc greedy search         | 6.87 % N=104765 C=98183 S=6408 D=174 I=620  |
| ctc prefix beam search    | 6.87 % N=104765 C=98189 S=6402 D=174 I=619  |
| attention rescoring       | 5.33 % N=104765 C=99354 S=5304 D=107 I=171  |
