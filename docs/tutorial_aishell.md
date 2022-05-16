## Tutorial on AIShell

If you meet any problems when going through this tutorial, please feel free to ask in github [issues](https://github.com/mobvoi/wenet/issues). Thanks for any kind of feedback.

### Setup environment

Please follow [Installation](https://github.com/wenet-e2e/wenet#installation) to install WeNet.

### First Experiment

We provide a recipe `example/aishell/s0/run.sh` on aishell-1 data.

The recipe is simple and we suggest you run each stage one by one manually and check the result to understand the whole process.

```
cd example/aishell/s0
bash run.sh --stage -1 --stop-stage -1
bash run.sh --stage 0 --stop-stage 0
bash run.sh --stage 1 --stop-stage 1
bash run.sh --stage 2 --stop-stage 2
bash run.sh --stage 3 --stop-stage 3
bash run.sh --stage 4 --stop-stage 4
bash run.sh --stage 5 --stop-stage 5
bash run.sh --stage 6 --stop-stage 6
```

You could also just run the whole script
```
bash run.sh --stage -1 --stop-stage 6
```


#### Stage -1: Download data

This stage downloads the aishell-1 data to the local path `$data`. This may take several hours. If you have already downloaded the data, please change the `$data` variable in `run.sh` and start from `--stage 0`.
Please set a **absolute path** for `$data`, e.g. `/home/username/asr-data/aishell/`

#### Stage 0: Prepare Training data

In this stage, `local/aishell_data_prep.sh` organizes the original aishell-1 data into two files:
* **wav.scp** each line records two tab-separated columns : `wav_id` and `wav_path`
* **text**  each line records two tab-separated columns :  `wav_id` and `text_label`

**wav.scp**
```
BAC009S0002W0122 /export/data/asr-data/OpenSLR/33/data_aishell/wav/train/S0002/BAC009S0002W0122.wav
BAC009S0002W0123 /export/data/asr-data/OpenSLR/33/data_aishell/wav/train/S0002/BAC009S0002W0123.wav
BAC009S0002W0124 /export/data/asr-data/OpenSLR/33/data_aishell/wav/train/S0002/BAC009S0002W0124.wav
BAC009S0002W0125 /export/data/asr-data/OpenSLR/33/data_aishell/wav/train/S0002/BAC009S0002W0125.wav
...
```

**text**
```
BAC009S0002W0122 而对楼市成交抑制作用最大的限购
BAC009S0002W0123 也成为地方政府的眼中钉
BAC009S0002W0124 自六月底呼和浩特市率先宣布取消限购后
BAC009S0002W0125 各地政府便纷纷跟进
...
```

If you want to train using your customized data, just organize the data into two files `wav.scp` and `text`, and start from `stage 1`.


#### Stage 1: Extract optinal cmvn features

`example/aishell/s0` uses raw wav as input and and [TorchAudio](https://pytorch.org/audio/stable/index.html) to extract the features just-in-time in dataloader. So in this step we just copy the training wav.scp and text file into the `raw_wav/train/` dir.

`tools/compute_cmvn_stats.py` is used to extract global cmvn(cepstral mean and variance normalization) statistics. These statistics will be used to normalize the acoustic features. Setting `cmvn=false` will skip this step.

#### Stage 2: Generate label token dictionary

The dict is a map between label tokens (we use characters for Aishell-1) and
 the integer indices.

An example dict is as follows
```
<blank> 0
<unk> 1
一 2
丁 3
...
龚 4230
龟 4231
<sos/eos> 4232
```

* `<blank>` denotes the blank symbol for CTC.
* `<unk>` denotes the unknown token, any out-of-vocabulary tokens will be mapped into it.
* `<sos/eos>` denotes start-of-speech and end-of-speech symbols for attention based encoder decoder training, and they shares the same id.

#### Stage 3: Prepare WeNet data format

This stage generates the WeNet required format file `data.list`. Each line in `data.list` is in json format which contains the following fields.

1. `key`: key of the utterance
2. `wav`: audio file path of the utterance
3. `txt`: normalized transcription of the utterance, the transcription will be tokenized to the model units on-the-fly at the training stage.

Here is an example of the `data.list`, and please see the generated training feature file in `data/train/data.list`.

```
{"key": "BAC009S0002W0122", "wav": "/export/data/asr-data/OpenSLR/33//data_aishell/wav/train/S0002/BAC009S0002W0122.wav", "txt": "而对楼市成交抑制作用最大的限购"}
{"key": "BAC009S0002W0123", "wav": "/export/data/asr-data/OpenSLR/33//data_aishell/wav/train/S0002/BAC009S0002W0123.wav", "txt": "也成为地方政府的眼中钉"}
{"key": "BAC009S0002W0124", "wav": "/export/data/asr-data/OpenSLR/33//data_aishell/wav/train/S0002/BAC009S0002W0124.wav", "txt": "自六月底呼和浩特市率先宣布取消限购后"}
```

We aslo design another format for `data.list` named `shard` which is for big data training.
Please see [gigaspeech](https://github.com/wenet-e2e/wenet/tree/main/examples/gigaspeech/s0)(10k hours) or
[wenetspeech](https://github.com/wenet-e2e/wenet/tree/main/examples/wenetspeech/s0)(10k hours)
for how to use `shard` style `data.list` if you want to apply WeNet on big data set(more than 5k).

#### Stage 4: Neural Network training

The NN model is trained in this step.

- Multi-GPU mode

If using DDP mode for multi-GPU, we suggest using `dist_backend="nccl"`. If the NCCL does not work, try using `gloo` or use `torch==1.6.0`
Set the GPU ids in CUDA_VISIBLE_DEVICES. For example, set `export CUDA_VISIBLE_DEVICES="0,1,2,3,6,7"` to use card 0,1,2,3,6,7.

- Resume training

If your experiment is terminated after running several epochs for some reasons (e.g. the GPU is accidentally used by other people and is out-of-memory ), you could continue the training from a checkpoint model. Just find out the finished epoch in `exp/your_exp/`, set  `checkpoint=exp/your_exp/$n.pt` and run the `run.sh --stage 4`. Then the training will continue from the $n+1.pt

- Config

The config of neural network structure, optimization parameter, loss parameters, and dataset can be set in a YAML format file.

In `conf/`,  we provide several models like transformer and conformer. see `conf/train_conformer.yaml` for reference.

- Use Tensorboard

The training takes several hours. The actual time depends on the number and type of your GPU cards. In an 8-card 2080 Ti machine, it takes about less than one day for 50 epochs.
You could use tensorboard to monitor the loss.

```
tensorboard --logdir tensorboard/$your_exp_name/ --port 12598 --bind_all
```

#### Stage 5: Recognize wav using the trained model

This stage shows how to recognize a set of wavs into texts. It also shows how to do the model averaging.

- Average model

If `${average_checkpoint}` is set to `true`, the best `${average_num}` models on cross validation set will be averaged to generate a boosted model and used for recognition.

- Decoding

Recognition is also called decoding or inference. The function of the NN will be applied on the input acoustic feature sequence to output a sequence of text.

Four decoding methods are provided in WeNet:

* `ctc_greedy_search` : encoder + CTC greedy search
* `ctc_prefix_beam_search` :  encoder + CTC prefix beam search
* `attention` : encoder + attention-based decoder decoding
* `attention_rescoring` : rescoring the ctc candidates from ctc prefix beam search with encoder output on attention-based decoder.

In general, attention_rescoring is the best method. Please see [U2 paper](https://arxiv.org/pdf/2012.05481.pdf) for the details of these algorithms.

`--beam_size` is a tunable parameter, a large beam size may get better results but also cause higher computation cost.

`--batch_size` can be greater than 1 for "ctc_greedy_search" and "attention" decoding mode, and must be 1 for "ctc_prefix_beam_search" and "attention_rescoring" decoding mode.

- WER evaluation

`tools/compute-wer.py` will calculate the word (or char) error rate of the result. If you run the recipe without any change, you may get WER ~= 5%.


#### Stage 6: Export the trained model

`wenet/bin/export_jit.py` will export the trained model using Libtorch. The exported model files can be easily used for inference in other programming languages such as C++.
