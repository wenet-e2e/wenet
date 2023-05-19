## Introduction
The below example shows how to deploy WeNet ASR offline or streaming models with hotwordsboosting on GPUs.

## Instructions
* Step 1. Convert your pretrained model to onnx models
```bash
conda activate wenet
pip install onnxruntime-gpu onnxmltools
cd wenet/examples/aishell2/s0 && . ./path.sh

# offline model
wget https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/aishell/20210601_u2%2B%2B_conformer_exp.tar.gz --no-check-certificate
tar -zxvf 20210601_u2++_conformer_exp.tar.gz
model_dir=$(pwd)/20210601_u2++_conformer_exp

# streaming model
wget https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/aishell/20210601_u2%2B%2B_conformer_exp.tar.gz --no-check-certificate
tar -zxvf 20210601_u2++_conformer_exp.tar.gz
model_dir=$(pwd)/20210601_u2++_conformer_exp

onnx_model_dir=$model_dir/onnx_model_dir
mkdir $onnx_model_dir

# offline model
python3 wenet/bin/export_onnx_gpu.py --config=$model_dir/train.yaml --checkpoint=$model_dir/final.pt --cmvn_file=$model_dir/global_cmvn --ctc_weight=0.3 --reverse_weight=0.3 --output_onnx_dir=$onnx_model_dir --fp16

# streaming model
python3 wenet/bin/export_onnx_gpu.py --config=$model_dir/train.yaml --checkpoint=$model_dir/final.pt --cmvn_file=$model_dir/global_cmvn --ctc_weight=0.3 --reverse_weight=0.3 --output_onnx_dir=$onnx_model_dir --fp16 --streaming

cp $model_dir/units.txt $onnx_model_dir
cp $model_dir/units.txt $onnx_model_dir/words.txt
cp $model_dir/train.yaml $onnx_model_dir/
```

* Step 2. Copy hotwords related files to onnx_model_dir folder
```bash
cd wenet/runtime/gpu/hotwords
cp hotwords.yaml $onnx_model_dir
```

* Step 3. Build server docker and start server on one gpu
```
cd wenet/runtime/gpu
# docker version >= 19.03.2
# If download fails, please use https://huggingface.co/58AILab/wenet_u2pp_aishell1_with_hotwords/blob/main/Dockerfile/Dockerfile.hotwordsserver
docker build . -f Dockerfile/Dockerfile.server -t wenet_hotwords_server:latest --network host

# offline model
docker run --gpus '"device=0"' --rm -it -v $PWD:/ws/gpu -v $PWD/hotwords/model_repo_hotwords:/ws/gpu/hotwords/model_repo -v $onnx_model_dir:/ws/onnx_model -p 8000:8000 -p 8001:8001 -p 8002:8002 --shm-size=1g --ulimit memlock=-1 wenet_hotwords_server:latest /workspace/scripts/convert_start_hotwords_server.sh

# streaming model
docker run --gpus '"device=0"' --rm -it -v $PWD:/ws/gpu -v $PWD/hotwords/model_repo_stateful_hotwords:/ws/gpu/hotwords/model_repo -v $onnx_model_dir:/ws/onnx_model -p 8000:8000 -p 8001:8001 -p 8002:8002 --shm-size=1g --ulimit memlock=-1  wenet_hotwords_server:latest /workspace/scripts/convert_start_hotwords_server.sh
```

* Step 4. Start client
```
cd wenet/runtime/gpu
docker build . -f Dockerfile/Dockerfile.client -t wenet_client:latest --network host
AUDIO_DATA=$PWD/client/test_wavs
docker run -it --net host --name wenet_hotwords_client -v $PWD/client:/ws/client -v $AUDIO_DATA:/ws/test_data wenet_client:latest

# In docker
# offline model test
cd /ws/client

# test one wav file
python3 client.py --audio_file=/ws/test_data/mid.wav --url=localhost:8001

# test a list of wav files & cer
python3 client.py --wavscp=/ws/dataset/test/wav.scp --data_dir=/ws/dataset/test/ --trans=/ws/dataset/test/text

# streaming model test
python3 client.py --audio_file=/ws/test_data/mid.wav --url=localhost:8001 --model_name=streaming_wenet --streaming
```

## Hotwords Test

Base Acoustic model: [20210601_u2++_conformer_exp (AISHELL-1)](https://github.com/wenet-e2e/wenet/blob/main/docs/pretrained_models.md)

Tested ENV
* CPU：40 Core, Intel(R) Xeon(R) Silver 4210 CPU @ 2.20GHz
* GPU：NVIDIA GeForce RTX 2080 Ti

Hotwords file: https://huggingface.co/58AILab/wenet_u2pp_aishell1_with_hotwords/tree/main/models

[AISHELL-1 Test dataset](https://www.openslr.org/33/)

* Test set contains 7176 utterances (5 hours) from 20 speakers.

| model (FP16)                 | RTF     | CER    |
|------------------------------|---------|--------|
| offline model w/o hotwords   | 0.00437 | 4.6805 |
| offline model w/  hotwords   | 0.00428 | 4.5841 |
| streaming model w/o hotwords | 0.01231 | 5.2777 |
| streaming model w/  hotwords | 0.01195 | 5.1850 |

[AISHELL-1 hostwords sub-testsets](https://www.modelscope.cn/datasets/speech_asr/speech_asr_aishell1_hotwords_testsets/summary)

* Test set contains 235 utterances with 187 entities words.

| model (FP16)               | Latency (s) | CER   | Recall | Precision | F1-score |
|----------------------------|-------------|-------|--------|-----------|----------|
| offline model w/o hotwords | 5.8673      | 13.85 | 0.27   | 0.99      | 0.43     |
| offline model w/  hotwords | 5.6601      | 11.96 | 0.47   | 0.97      | 0.63     |

Decoding result

| Label                | hotwords  | pred w/o hotwords            | pred w/ hotwords             |
|----------------------|-----------|------------------------------|------------------------------|
| 以及拥有陈露的女单项目          | 陈露        | 以及拥有**陈鹭**的女单项目              | 以及拥有**陈露**的女单项目              |
| 庞清和佟健终于可以放心地考虑退役的事情了 | 庞清<br/>佟健 | **庞青**和**董建**终于可以放心地考虑退役的事情了 | **庞清**和**佟健**终于可以放心地考虑退役的事情了 |
| 赵继宏老板电器做厨电已经三十多年了    | 赵继宏       | **赵继红**老板电器做厨店已经三十多年了        | **赵继宏**老板电器做厨电已经三十多年了        |

Refer to more results: https://huggingface.co/58AILab/wenet_u2pp_aishell1_with_hotwords/tree/main/results

## Hotwords usage
Please refer to the following steps how to use hotwordsboosting.
* Step 1. Initialize HotWordsScorer
```
# if you don't want to use hotwords. set hotwords_scorer=None(default),
# vocab_list is Chinese characters.
hot_words = {'再接': 10, '再厉': -10, '好好学习': 100}
hotwords_scorer = HotWordsScorer(hot_words, vocab_list, is_character_based=True)
```
If you set is_character_based is True (default mode), the first step is to combine Chinese characters into words, if words in hotwords dictionary then add hotwords score. If you set is_character_based is False, all words in the fixed window will be enumerated.

* Step 2. Add hotwords_scorer when decoding
```
result = ctc_beam_search_decoder_batch(batch_chunk_log_prob_seq,
                                        batch_chunk_log_probs_idx,
                                        batch_root_trie,
                                        batch_start,
                                        beam_size, num_processes,
                                        blank_id, space_id,
                                        cutoff_prob, scorer, hotwords_scorer)
```
Please refer to [swig/test/test_zh.py](https://github.com/Slyne/ctc_decoder/blob/master/swig/test/test_zh.py#L108) for how to decode with hotwordsboosting.
