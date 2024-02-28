# Preliminary
1. Download whisper ckpt from this [link](https://github.com/openai/whisper/blob/main/whisper/__init__.py#L17-L30)

2. We assume you have run stage0~stage3 using `wenetspeech/s0/run.sh` and here we are simply creating a symbolic link
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
# remove conv1d2
python local/modify_ckpt.py \
  --remove_list "encoder.embed.conv" \
  --input_ckpt exp/whisper/large-v3/wenet_whisper.pt \
  --output_ckpt exp/whisper/large-v3/wenet_whisper.remove-subsample.pt
# init ctc
python local/modify_ckpt.py \
  --add_list "{\"ctc.ctc_lo.weight\": \"decoder.embed.0.weight\"}" \
  --input_ckpt exp/whisper/large-v3/wenet_whisper.pt \
  --output_ckpt exp/whisper/large-v3/wenet_whisper.init-ctc.pt
# remove conv1d2 and init ctc
python local/modify_ckpt.py \
  --remove_list "encoder.embed.conv" \
  --add_list "{\"ctc.ctc_lo.weight\": \"decoder.embed.0.weight\"}" \
  --input_ckpt exp/whisper/large-v3/wenet_whisper.pt \
  --output_ckpt exp/whisper/large-v3/wenet_whisper.remove-subsample.init-ctc.pt
```

# Performance Record

## Whisper-largev2 (original) Result

|   decoding_method   |  Dev | Test\_Net | Test\_Meeting |
|:-------------------:|:----:|:---------:|:-------------:|
|  ctc_greedy_search  | N/A  |   N/A     |     N/A     |
|      attention      | N/A  |  10.99    |   26.32     |
| attention_rescoring | N/A  |    N/A    |     N/A     |

## Whisper-largev3 (conv1d2, full-parameter tuning) Result

* Feature info: using log_mel_spectrogram feature, no cmvn
* Training info: bf16, deepspeed stage1, activation checkpointing, batch dynamic12000, acc_grad 8, 8 * 3090 gpu, 40k steps (about 5 days), conf/finetune_whisper_largev3.yaml
* Decoding info: ctc_weight 0.0, average_num 5
* PR link: https://github.com/wenet-e2e/wenet/pull/2356

|   decoding_method   |  Dev | Test\_Net | Test\_Meeting |
|:-------------------:|:----:|:---------:|:-------------:|
|  ctc_greedy_search  | 10.78 % N=328207 C=305525 S=18448 D=4234 I=12695  |   12.52 % N=414097 C=366245 S=34946 D=12906 I=4004     |    17.74 % N=220358 C=193382 S=20284 D=6692 I=12108     |
|      attention      | 7.27 % N=328207 C=308016 S=11392 D=8799 I=3672  |  7.90 % N=414097 C=383382 S=18954 D=11761 I=2018    |   13.00 % N=220358 C=194417 S=11788 D=14153 I=2705     |
| attention_rescoring | 8.95 % N=328207 C=305892 S=16696 D=5619 I=7056  |    10.83 % N=414097 C=371515 S=30229 D=12353 I=2269    |    15.64 % N=220358 C=193717 S=18669 D=7972 I=7812     |

## Whisper-largev3 (conv1d2, full-parameter tuning) Result (text\_fixed, see https://github.com/wenet-e2e/WenetSpeech/discussions/54)

* Feature info: using log_mel_spectrogram feature, no cmvn
* Training info: bf16, deepspeed stage1, activation checkpointing, batch dynamic12000, acc_grad 8, 8 * 3090 gpu, 48k steps (about 6 days), conf/finetune_whisper_largev3.yaml
* Decoding info: ctc_weight 0.0, average_num 5
* PR link: https://github.com/wenet-e2e/wenet/pull/2371

|   decoding_method   |  Dev | Test\_Net | Test\_Meeting |
|:-------------------:|:----:|:---------:|:-------------:|
|  ctc_greedy_search  | 7.09 % N=328207 C=308643 S=16976 D=2588 I=3709  | 10.98 % N=414092 C=373301 S=33375 D=7416 I=4697 | 12.84 % N=220358 C=194928 S=18398 D=7032 I=2862 |
|      attention      | 4.66 % N=328207 C=315591 S=10352 D=2264 I=2692  | 6.54 % N=414092 C=389523 S=19101 D=5468 I=2513 | 8.84 % N=220358 C=202722 S=11296 D=6340 I=1839  |
| attention_rescoring | 5.99 % N=328207 C=311106 S=14807 D=2294 I=2547  | 9.27 % N=414092 C=378406 S=28993 D=6693 I=2715 | 11.47 % N=220358 C=197013 S=16716 D=6629 I=1923 |

# Frequently Asked Questions

- Q: Why are there so many insertion errors in the decoding results of CTC and attention_rescoring?
- A: Because Chinese characters are composed of multiple bytes, in Whisper's tokenizer, one Chinese character might be represented by multiple tokens (for example, 3 tokens). During the CTC decoding process, it's possible that two or four tokens are decoded. This not only causes garbled text (see https://github.com/wenet-e2e/wenet/issues/2308 ) but also leads to insertion errors.
