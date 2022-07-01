## Tutorial on LibriSpeech

If you meet any problems when going through this tutorial, please feel free to ask in github [issues](https://github.com/mobvoi/wenet/issues). Thanks for any kind of feedback.

### Setup environment

Please follow [Installation](https://github.com/wenet-e2e/wenet#installation) to install WeNet.

### First Experiment

We provide a recipe `example/librispeech/s0/run.sh` on librispeech data.

The recipe is simple and we suggest you run each stage one by one manually and check the result to understand the whole process.

```
cd example/librispeech/s0
bash run.sh --stage -1 --stop-stage -1
bash run.sh --stage 0 --stop-stage 0
bash run.sh --stage 1 --stop-stage 1
bash run.sh --stage 2 --stop-stage 2
bash run.sh --stage 3 --stop-stage 3
bash run.sh --stage 4 --stop-stage 4
bash run.sh --stage 5 --stop-stage 5
bash run.sh --stage 6 --stop-stage 6
bash run.sh --stage 7 --stop-stage 7
```

You could also just run the whole script
```
bash run.sh --stage -1 --stop-stage 7
```


#### Stage -1: Download data

``` sh
data_url=www.openslr.org/resources/12
datadir=/export/data/en-asr-data/OpenSLR/
if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
  echo "stage -1: Data Download"
  for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
    local/download_and_untar.sh ${datadir} ${data_url} ${part}
  done
fi
```

This stage downloads the librispeech data to the local path `$data`. This may take several hours. If you have already downloaded the data, please change the `$data` variable in `run.sh` and start from `--stage 0`.

#### Stage 0: Prepare Training data

``` sh
wave_data=data
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
    # use underscore-separated names in data directories.
    local/data_prep_torchaudio.sh ${datadir}/LibriSpeech/${part} $wave_data/${part//-/_}
  done
fi
```

In this stage, `local/data_prep_torchaudio.sh` organizes the original data into two files:
* **wav.scp** each line records two tab-separated columns : `wav_id` and `wav_path`
* **text**  each line records two tab-separated columns :  `wav_id` and `text_label`

**wav.scp**
```
1867-154075-0014 /export/data/en-asr-data/OpenSLR//LibriSpeech/train-clean-100/1867/154075/1867-154075-0014.flac
1970-26100-0022 /export/data/en-asr-data/OpenSLR//LibriSpeech/train-clean-100/1970/26100/1970-26100-0022.flac
...
```

**text**
```
1867-154075-0014 YOU SHOW HIM THAT IT IS POSSIBLE
1970-26100-0022 DID YOU SEE HIM AT THAT TIME
...
```

If you want to train using your customized data, just organize the data into two files `wav.scp` and `text`, and start from `stage 1`.


#### Stage 1: Extract optinal cmvn features

``` sh
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  ### Task dependent. You have to design training and dev sets by yourself.
  ### But you can utilize Kaldi recipes in most cases
  echo "stage 1: Feature Generation"
  mkdir -p $wave_data/train_960
  # merge total training data
  for set in train_clean_100 train_clean_360 train_other_500; do
    for f in `ls $wave_data/$set`; do
      cat $wave_data/$set/$f >> $wave_data/train_960/$f
    done
  done
  mkdir -p $wave_data/dev
  # merge total dev data
  for set in dev_clean dev_other; do
    for f in `ls $wave_data/$set`; do
      cat $wave_data/$set/$f >> $wave_data/dev/$f
    done
  done

  tools/compute_cmvn_stats.py --num_workers 16 --train_config $train_config \
    --in_scp $wave_data/$train_set/wav.scp \
    --out_cmvn $wave_data/$train_set/global_cmvn

fi
```

The librispeech corpus contains 3 subsets for training, namely `train_clean_100`, `train_clean_360`, and `train_other_500`,
so we first merge them to get our final training data.

`tools/compute_cmvn_stats.py` is used to extract global cmvn(cepstral mean and variance normalization) statistics. These statistics will be used to normalize the acoustic features. Setting `cmvn=false` will skip this step.

#### Stage 2: Generate label token dictionary

``` sh
dict=$wave_data/lang_char/${train_set}_${bpemode}${nbpe}_units.txt
bpemodel=$wave_data/lang_char/${train_set}_${bpemode}${nbpe}
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  ### Task dependent. You have to check non-linguistic symbols used in the corpus.
  echo "stage 2: Dictionary and Json Data Preparation"
  mkdir -p data/lang_char/

  echo "<blank> 0" > ${dict} # 0 will be used for "blank" in CTC
  echo "<unk> 1" >> ${dict} # <unk> must be 1

  # we borrowed these code and scripts which are related bpe from ESPnet.
  cut -f 2- -d" " $wave_data/${train_set}/text > $wave_data/lang_char/input.txt
  tools/spm_train --input=$wave_data/lang_char/input.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000
  tools/spm_encode --model=${bpemodel}.model --output_format=piece < $wave_data/lang_char/input.txt | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+1}' >> ${dict}
  num_token=$(cat $dict | wc -l)
  echo "<sos/eos> $num_token" >> $dict # <eos>
  wc -l ${dict}
fi
```

The model unit of English e2e speech recognition system could be char or BPE(byte-pair-encoding).
Typically, BPE shows better result. So here we use BPE as model unit,
and the BPE is trained by [sentencepiece](https://github.com/google/sentencepiece) tool on the librispeech training data.

The model unit is defined as a dict in WeNet, which maps the a BPE into integer index.
The librispeech dict is like:

```
<blank> 0
<unk> 1
' 2
▁ 3
A 4
▁A 5
AB 6
▁AB 7
▁YOU 4995
▁YOUNG 4996
▁YOUR 4997
▁YOUTH 4998
Z 4999
ZZ 5000
<sos/eos> 5001
```

* `<blank>` denotes the blank symbol for CTC.
* `<unk>` denotes the unknown token, any out-of-vocabulary tokens will be mapped into it.
* `<sos/eos>` denotes start-of-speech and end-of-speech symbols for attention based encoder decoder training, and they shares the same id.

#### Stage 3: Prepare WeNet data format

``` sh
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  # Prepare wenet requried data
  echo "Prepare data, prepare requried format"
  for x in dev ${recog_set} $train_set ; do
    tools/make_raw_list.py $wave_data/$x/wav.scp $wave_data/$x/text \
        $wave_data/$x/data.list
  done

fi
```

This stage generates the WeNet required format file `data.list`. Each line in `data.list` is in json format which contains the following fields.

1. `key`: key of the utterance
2. `wav`: audio file path of the utterance
3. `txt`: normalized transcription of the utterance, the transcription will be tokenized to the model units on-the-fly at the training stage.

Here is an example of the `data.list`, and please see the generated training feature file in `data/train/data.list`.

```
{"key": "1455-134435-0000", "wav": "/mnt/nfs/ptm1/open-data/LibriSpeech/train-clean-100/1455/134435/1455-134435-0000.flac", "txt": "THE GIRL WHO CAME INTO THE WORLD ON THAT NIGHT WHEN JESSE RAN THROUGH THE FIELDS CRYING TO GOD THAT HE BE GIVEN A SON HAD GROWN TO WOMANHOOD ON THE FARM"}
{"key": "1455-134435-0001", "wav": "/mnt/nfs/ptm1/open-data/LibriSpeech/train-clean-100/1455/134435/1455-134435-0001.flac", "txt": "AND WHEN NOT ANGRY SHE WAS OFTEN MOROSE AND SILENT IN WINESBURG IT WAS SAID THAT SHE DRANK HER HUSBAND THE BANKER"}
{"key": "1455-134435-0002", "wav": "/mnt/nfs/ptm1/open-data/LibriSpeech/train-clean-100/1455/134435/1455-134435-0002.flac", "txt": "BUT LOUISE COULD NOT BE MADE HAPPY SHE FLEW INTO HALF INSANE FITS OF TEMPER DURING WHICH SHE WAS SOMETIMES SILENT SOMETIMES NOISY AND QUARRELSOME SHE SWORE AND CRIED OUT IN HER ANGER SHE GOT A KNIFE FROM THE KITCHEN AND THREATENED HER HUSBAND'S LIFE"}
```

We aslo design another format for `data.list` named `shard` which is for big data training.
Please see [gigaspeech](https://github.com/wenet-e2e/wenet/tree/main/examples/gigaspeech/s0)(10k hours) or
[wenetspeech](https://github.com/wenet-e2e/wenet/tree/main/examples/wenetspeech/s0)(10k hours)
for how to use `shard` style `data.list` if you want to apply WeNet on big data set(more than 5k).

#### Stage 4: Neural Network training

``` sh
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  # Training
  mkdir -p $dir
  INIT_FILE=$dir/ddp_init
  rm -f $INIT_FILE # delete old one before starting
  init_method=file://$(readlink -f $INIT_FILE)
  echo "$0: init method is $init_method"
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  # Use "nccl" if it works, otherwise use "gloo"
  dist_backend="nccl"
  cmvn_opts=
  $cmvn && cmvn_opts="--cmvn $wave_data/${train_set}/global_cmvn"
  # train.py will write $train_config to $dir/train.yaml with model input
  # and output dimension, train.yaml will be used for inference or model
  # export later
  for ((i = 0; i < $num_gpus; ++i)); do
  {
    gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$[$i+1])
    python wenet/bin/train.py --gpu $gpu_id \
      --config $train_config \
      --data_type raw \
      --symbol_table $dict \
      --train_data $wave_data/$train_set/data.list \
      --cv_data $wave_data/dev/data.list \
      ${checkpoint:+--checkpoint $checkpoint} \
      --model_dir $dir \
      --ddp.init_method $init_method \
      --ddp.world_size $num_gpus \
      --ddp.rank $i \
      --ddp.dist_backend $dist_backend \
      --num_workers 1 \
      $cmvn_opts \
      --pin_memory
  } &
  done
  wait
fi
```

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

``` sh
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
# Test model, please specify the model you want to test by --checkpoint
  cmvn_opts=
  $cmvn && cmvn_opts="--cmvn data/${train_set}/global_cmvn"
  # TODO, Add model average here
  mkdir -p $dir/test
  if [ ${average_checkpoint} == true ]; then
    decode_checkpoint=$dir/avg_${average_num}.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python wenet/bin/average_model.py \
      --dst_model $decode_checkpoint \
      --src_path $dir  \
      --num ${average_num} \
      --val_best
  fi
  # Specify decoding_chunk_size if it's a unified dynamic chunk trained model
  # -1 for full chunk
  decoding_chunk_size=
  ctc_weight=0.5
  # Polling GPU id begin with index 0
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  idx=0
  for test in $recog_set; do
    for mode in ${decode_modes}; do
    {
      {
        test_dir=$dir/${test}_${mode}
        mkdir -p $test_dir
        gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$[$idx+1])
        python wenet/bin/recognize.py --gpu $gpu_id \
          --mode $mode \
          --config $dir/train.yaml \
          --data_type raw \
          --test_data $wave_data/$test/data.list \
          --checkpoint $decode_checkpoint \
          --beam_size 10 \
          --batch_size 1 \
          --penalty 0.0 \
          --dict $dict \
          --result_file $test_dir/text_bpe \
          --ctc_weight $ctc_weight \
          ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}

        tools/spm_decode --model=${bpemodel}.model --input_format=piece < $test_dir/text_bpe | sed -e "s/▁/ /g" > $test_dir/text
        python tools/compute-wer.py --char=1 --v=1 \
          $wave_data/$test/text $test_dir/text > $test_dir/wer
      } &

      ((idx+=1))
      if [ $idx -eq $num_gpus ]; then
        idx=0
      fi
    }
    done
  done
  wait

fi

```

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

`tools/compute-wer.py` will calculate the word (or char) error rate of the result.


#### Stage 6(Optional): Export the trained model

``` sh
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  # Export the best model you want
  python wenet/bin/export_jit.py \
    --config $dir/train.yaml \
    --checkpoint $dir/avg_${average_num}.pt \
    --output_file $dir/final.zip
fi
```

`wenet/bin/export_jit.py` will export the trained model using Libtorch.
The exported model files can be easily used for C++ inference in our runtime.
It is required if you want to integrate language model(LM), as shown in Stage 7.


#### Stage 7(Optional): Add LM and test it with runtime



``` sh
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  lm=data/local/lm
  lexicon=data/local/dict/lexicon.txt
  mkdir -p $lm
  mkdir -p data/local/dict

  # 7.1 Download & format LM
  which_lm=3-gram.pruned.1e-7.arpa.gz
  if [ ! -e ${lm}/${which_lm} ]; then
    wget http://www.openslr.org/resources/11/${which_lm} -P ${lm}
  fi
  echo "unzip lm($which_lm)..."
  gunzip -k ${lm}/${which_lm} -c > ${lm}/lm.arpa
  echo "Lm saved as ${lm}/lm.arpa"

  # 7.2 Prepare dict
  unit_file=$dict
  bpemodel=$bpemodel
  # use $dir/words.txt (unit_file) and $dir/train_960_unigram5000 (bpemodel)
  # if you download pretrained librispeech conformer model
  cp $unit_file data/local/dict/units.txt
  if [ ! -e ${lm}/librispeech-lexicon.txt ]; then
    wget http://www.openslr.org/resources/11/librispeech-lexicon.txt -P ${lm}
  fi
  echo "build lexicon..."
  tools/fst/prepare_dict.py $unit_file ${lm}/librispeech-lexicon.txt \
    $lexicon $bpemodel.model
  echo "lexicon saved as '$lexicon'"

  # 7.3 Build decoding TLG
  tools/fst/compile_lexicon_token_fst.sh \
     data/local/dict data/local/tmp data/local/lang
  tools/fst/make_tlg.sh data/local/lm data/local/lang data/lang_test || exit 1;

  # 7.4 Decoding with runtime
  fst_dir=data/lang_test
  for test in ${recog_set}; do
    ./tools/decode.sh --nj 6 \
      --beam 10.0 --lattice_beam 5 --max_active 7000 --blank_skip_thresh 0.98 \
      --ctc_weight 0.5 --rescoring_weight 1.0 --acoustic_scale 1.2 \
      --fst_path $fst_dir/TLG.fst \
      --dict_path $fst_dir/words.txt \
      data/$test/wav.scp data/$test/text $dir/final.zip $fst_dir/units.txt \
      $dir/lm_with_runtime_${test}
    tail $dir/lm_with_runtime_${test}/wer
  done
fi
```

LM is  only supported in runtime, you have to build the runtime as shown in [Installation](https://github.com/wenet-e2e/wenet#installation),
and please refer [LM for WeNet](https://wenet-e2e.github.io/wenet/lm.html) for the details of LM design.


