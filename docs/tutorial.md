## Tutorial

If you meet any problems when going through this tutorial, please feel free to ask in github [issues](https://github.com/mobvoi/wenet/issues). And thanks for any kind feedbacks.

### Setup environment
- Clone

```sh
git clone https://github.com/mobvoi/wenet.git
```



- Install Conda

https://docs.conda.io/en/latest/miniconda.html


- Create conda env

pytorch 1.6.0 is suggested. We meet some error on NCCL when using 1.7.0 on 2080 Ti.

```
conda create -n wenet python=3.8
conda activate wenet
pip install -r requirements.txt
conda install pytorch==1.6.0 cudatoolkit=10.1 -c pytorch
```

- Install Kaldi

Wenet need Kaldi to extract feature (a torchaudio version is in developing)

Download and build [Kaldi](https://github.com/kaldi-asr/kaldi).

Set the Kaldi root
```
vim example/aishell/s0/path.sh
KALDI_ROOT=your_kaldi_root_path
```


### First Experiment

We provide a recipe example/aishell/s0/run.sh on aishell-1 data.

The recipe is simple and we suggest you run each stage one by one manually and check the result to understand the whole process.

```
cd example/aishell/s0
bash run.sh --stage -1 --stop-stage -1
bash run.sh --stage 0 --stop-stage 0
bash run.sh --stage 1 --stop-stage 1
bash run.sh --stage 2 --stop-stage 2
bash run.sh --stage 3 --stop-stage 3
bash run.sh --stage 4 --stop-stage 5
bash run.sh --stage 5 --stop-stage 5
```

You could also just run the whole script
```
bash run.sh --stage -1 --stop-stage 5
```



#### stage -1 Download data

This stage downloads the data to the local path `$data`. This may take several hours.  If you already downloaded the aishell-1 data. please change the `$data` variable in `run.sh` and start from `--stage 0`.

#### stage 0 Prepare Kaldi format

In this stage, `local/aishell_data_prep.sh` organizes the original aishell-1 data into a Kaldi format. In the kaldi dataset format, each dataset includes the two most important files `wav.scp`, `text`, and other resource files.

In this demo recipe, this stage also uses Kaldi script `utils/perturb_data_dir_speed.sh` to do speed perturb augmentation on the training data. With 0.9 and 1.1 speeding up, the size of the training data is 3 times of the original data.

please see `data/train/wav.scp` and `data/train/text`. 
If you want to train your customized data, just organize the data into two files wav.scp and text, and start from `stage 1`.


#### stage 1 Extract acoustic feature

In this stage, Kaldi script `steps/make_fbank_pitch.sh` is used to extract the acoustic feature from the wavs for train, test, and dev set.

`compute-cmvn-stats` is used to extract global cmvn statistics. These statistics will be used to normalize the acoustic feature. Set `cmvn=false` will ignore this step.

See the generated training feature file in `fbank_pitch/train/feat.scp`.

#### stage 2 Generate label token dictionary

The dict, a map between label tokens(we use the character for Aishell-1) and
 the integer indexes.


A dict is like this
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

* `<blank>` denotes blank symbol for CTC.
* `<unk>` denotes unknown token, any out of vocabulary token will be mapped into .
* `<sos/eos>` denotes sos and eos symbol for attention encoder decoder training, and they shares the same id.

#### stage 3 Prepare Wenet data format

This stage generates a single wenet format file including all the input/output information needed by neural network training/evaluation.

See the generated training feature file in `fbank_pitch/train/format.data`.


In the wenet format file , each line record a data sample of seven tab splited columns. For example, a line is like this(repalce tab with newline here):

```
utt:BAC009S0764W0121
feat:/l/wenet/examples/aishell/s0/fbank_pitch/test/data/raw_fbank_pitch_test.1.ark:17
feat_shape:418,83
text:甚至出现交易几乎停滞的情况
token:甚 至 出 现 交 易 几 乎 停 滞 的 情 况
tokenid:2474 3116 331 2408 82 1684 321 47 235 2199 2553 1319 307
token_shape:13,4233
```

In the future, wenet will also support using raw wavs as input and use torchaudio to extract the feature just-in-time in dataloader.


#### stage 4 Neural Network training

The NN model is training in this step.

- Multi-GPU mode

If using DDP mode for multi-GPU, we suggest using `dist_backend="nccl"`. If the NCCL does not work, try using `gloo` or use `torch==1.6.0`
Set the GPU ids in CUDA_VISIBLE_DEVICES. For example, set `export CUDA_VISIBLE_DEVICES="0,1,2,3,6,7"` to use card 0,1,2,3,6,7.

- Resume training

If your experiment is terminated after running several epochs for some reasons(e.g. the GPU is accidentally used by other people and is out-of-memory ), you could continue the training on a checkpoint model. Just check the finished epoch in `exp/your_exp/` and set  `checkpoint=exp/your_exp/$n.pt` and run the `run.sh --stage 4`. Then the training will continue from the $n+1.pt

- Config

The config of neural network structure, optimization parameter, loss parameters, and dataset could be set in a YAML format file.

In `conf/`,  we provide several models like transformer and conformer. see `conf/train_conformer.yaml` for reference.

- Use Tensorboard

The training takes several hours. The actual hours depend on your GPU number and type. In an 8 card 2080 Ti machine, it will take about less than one day for 50 epochs.
You could use the tensorboard to monitor the loss.

```
tensorboard --logdir tensorboard/$your_exp_name/ --port 12598 --bind_all
```

#### stage 5 Recognize wav using the trained model

This stage shows how to recognize a set of wavs into text. It also shows how to do the model average.

- Average model

if `${average_checkpoint}` is `true`, the best `${average_num}` model on cross validation set will be averaged to generated a boosted model and used for recognition.

- Decoding

Recognition is also called decoding or inference. The function of the NN will be applied on the input acoustic feature sequence to out a text sequence.

Four decoding methods are provided in wenet:

* `ctc_greedy_search` : encoder + CTC greedy search
* `ctc_prefix_beam_search` :  encoder + CTC prefix beam search
* `attention` : encoder + attention-based decoder decoding
* `attention_rescoring` : rescoring the ctc candidates from ctc prefix beam search with encoder output on attention-based decoder.

In general, attention_rescoring is the best method. Please see [U2 paper](https://arxiv.org/pdf/2012.05481.pdf) for the details of these algorithms.

`--beam_size` is a tunable parameter, use a large beam size may get better results but also cause high computation cost.

`--batch_size` can be greater than 1 for "ctc_greedy_search" and "attention" decoding mode, and must be 1 for "ctc_prefix_beam_search" and "attention_rescoring" decoding mode.

- WER Evaluation

`tools/compute-wer.py` will calculate the word(or char) error rate of the result. If you run the recipe without any change, you may get WER ~= 5%.


#### stage 6 Export the trained model

`wenet/bin/export_jit.py` will export the trained model using libtorch. The exported model files could be easily used in other language appliction such as C++.