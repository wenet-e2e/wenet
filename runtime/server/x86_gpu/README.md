## Introduction
The below example shows how to deploy WeNet offline and online ASR models on GPUs.

## Instructions
* Step 1. Convert your model/pretrained model to onnx models. For example:

```bash
conda activate wenet
pip install onnxruntime-gpu onnxmltools
cd wenet/examples/aishell2 && . ./path.sh
model_dir=<absolute path to>/20211025_conformer_exp
onnx_model_dir=<absolute path>
mkdir $onnx_model_dir
python3 wenet/bin/export_onnx_gpu.py --config=$model_dir/train.yaml --checkpoint=$model_dir/final.pt --cmvn_file=$model_dir/global_cmvn --ctc_weight=0.5 --output_onnx_dir=$onnx_model_dir --fp16
cp $model_dir/units.txt $model_dir/train.yaml $onnx_model_dir/
```

If you want to export streaming model (u2/u2++) for streaming inference (inference by chunks) instead of offline inference (inference by audio segments/utterance), you should add `--streaming` option:

```
python3 wenet/bin/export_onnx_gpu.py --config=$model_dir/train.yaml --checkpoint=$model_dir/final.pt --cmvn_file=$model_dir/global_cmvn  --ctc_weight=0.1 --reverse_weight=0.4 --output_onnx_dir=$onnx_model_dir --fp16 --streaming
```

Congratulations! You've successully exported the onnx models and now you are able to deploy them. Please also ensure you have set up [NGC](https://catalog.ngc.nvidia.com/) account before the next step!

* Step 2. Build server docker and start server on one gpu:

```
docker build . -f Dockerfile/Dockerfile.server -t wenet_server:latest --network host
# offline model
docker run --gpus '"device=0"' -it -v $PWD/model_repo:/ws/model_repo -v $onnx_model_dir:/ws/onnx_model -p 8000:8000 -p 8001:8001 -p 8002:8002 --shm-size=1g --ulimit memlock=-1  wenet_server:latest /workspace/scripts/convert_start_server.sh
# streaming model
docker run --gpus '"device=0"' -it -v $PWD/model_repo_stateful:/ws/model_repo -v $onnx_model_dir:/ws/onnx_model -p 8000:8000 -p 8001:8001 -p 8002:8002 --shm-size=1g --ulimit memlock=-1  wenet_server:latest /workspace/scripts/convert_start_server.sh
```
Whenever there's something wrong starting the server, you may check the config.pbtxt files in every model in `model_repo` or `model_repo_stateful` to find if the settings are right.

* Step 3. Start client

```
docker build . -f Dockerfile/Dockerfile.client -t wenet_client:latest --network host

AUDIO_DATA=<path to your wav data>
docker run -ti --net host --name wenet_client -v $PWD/client:/ws/client -v $AUDIO_DATA:/ws/test_data wenet_client:latest
# In docker
# offline model test
cd /ws/client
# test one wav file
python3 client.py --audio_file=/ws/test_data/mid.wav --url=localhost:8001

# test a list of wav files & cer
python3 client.py --wavscp=/ws/dataset/test/wav.scp --data_dir=/ws/dataset/test/ --trans=/ws/dataset/test/text
```

Similarly, if your model is exported with `--streaming`, you should add this option when calling your streaming model.
For example,

```
python3 client.py --wavscp=/ws/test_data/data_aishell2/test/wav.scp --data_dir=/ws/test_data/data_aishell2/test/ --trans=/ws/test_data/data_aishell2/test/trans.txt --model_name=streaming_wenet --streaming
```

<img src="test.gif" alt="test" width="500"/>

## Precision Impact
Some of you may be worried about whether fp16 will affect the final accuracy. We did several experiments and we may find the accuracy is acceptable.
|Model | Dataset | Precision | CER |
|------------|-----------|-----------|----------|
|Aishell2-U2++ Conformer|Aishell2-TEST|FP16| 5.39%|
|Aishell2-U2++ Conformer|Aishell2-TEST|FP32| 5.38%|
|Wenetspeech Conformer| Wenetspeech-DEV|FP16| 8.61%|
|Wenetspeech Conformer| Wenetspeech-DEV|FP32| 8.61%|
|Wenetspeech Conformer | Wenetspeech-TestNet|FP16|9.07%|
|Wenetspeech Conformer| Wenetspeech-TestNet|FP32|9.06%|
|Wenetspeech Concofmer| Wenetspeech-TestMeeting|FP16|15.72%|
|Wenetspeech Conformer| Wenetspeech-TestMeeting|FP32|15.70%|


## Perf Test
We use the below command to do our testing and we run the below command several times to warm up:

```
cd /ws/client
# generate the test data, input to our feature extractor
# offline model
python3 generate_perf_input.py --audio_file=input.wav
# offline_input.json generated
perf_analyzer -m attention_rescoring -b 1 -p 20000 --concurrency-range 100:200:50 -i gRPC --input-data=offline_input.json  -u localhost:8001

# streaming input
python3 generate_perf_input.py --audio_file=input.wav --streaming
# online_input.json generated
perf_analyzer -u "localhost:8001" -i gRPC --streaming --input-data=online_input.json -m streaming_wenet -b 1 --concurrency-range 100:200:50
```
Where:
- `input.wav` the input test audio, we tested 5 seconds, 8 seconds, 10 seconds audio;
- `-m` option indicates the name of the served model;
- `-p` option is the mearsurement window, which indicates in what time duration to collect the metrics;
- `-v` option turns on the verbose model;
- `-i` option is for choosing the networking protocol, you can choose `HTTP` or `gRPC` here;
- `-u` option sets the url of the service in the form of `<IP Adrress>:<Port>`, but notice that port `8000` corresponds to HTTP protocol while port `8001` corresponds to gRPC protocol;
- `-b` option indicates the batch size of the input requests used fo testing; since we simulate individual users sending requests, we set batch size here to `1`;
- `--input-data` option points to the path of the json file containing the real input data
- `--concurrency-range` option is an important one, it indicates the concurrency of the requests which defines the pressure we will give to the server.
- You can also set `-f` option to set the path of testing result file;
- You can also set `--max-threads` option to set the number of threads used to send test request, it should be set to the number of CPU cores in your test machine.

### Tested ENV
* NVIDIA DRIVER: 470.57.02
* GPU: V100 & T4 & A30
* CUDA: 11.4
* Triton Inference Server: 22.03

### Offline Model Perf
Here are the wenetspeech conformer model and aishell2 u2++ perf on T4.
* Aishell2, FP16, onnx, [U2++ Conformer](http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/aishell2/20210618_u2pp_conformer_exp.tar.gz)

|input length|Concurrency|RTF / GPU  |Throughput|Latency_p50 (ms)|Latency_p90 (ms)|Latency_p95 (ms)|Latency_p99 (ms)|
|------------|-----------|-----------|----------|----------------|----------------|----------------|----------------|
|5s          | 50        |0.0010     |204       |245.003         |277.086         |284.818         |295.777         |
|5s          | 100       |0.0009     |225       | 452.578        |492.355         |506.399         |533.325         |
|5s          | 150       |0.0009     |228       |657.478         |722.427         |747.493         |794.346         |
|5s          |200        |0.0009     |228       |875.721         |946.02          |975.703         |1020.615        |
|10s         |10         |0.0011     |88        |116.203         |113.929         |136.902         |149.18          |
|10s         |50         |0.0009     |108       |476.678         |522.65          |532.693         |562.097         |
|10s         |100        |0.0009     |110       |921.383         |1001.848        |1029.96         |1067.6          |

* Wenetspeech, FP16, onnx, [Conformer](http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/wenetspeech/20211025_conformer_exp.tar.gz)

|input length|Concurrency|RTF / GPU  |Throughput|Latency_p50 (ms)|Latency_p90 (ms)|Latency_p95 (ms)|Latency_p99 (ms)|
|------------|-----------|-----------|----------|----------------|----------------|----------------|----------------|
|5s          |50         |0.0018     |110       |464.18          |508.246         |517.967         |547.002         |
|5s          |100        |0.0018     |112       |891.173         |958.046         |1011.058        |1093.231        |
|10s         |5          |0.0020     |50        |100.35          |120.757         |122.148         |132.053         |
|10s         |10         |0.0020     |50        |201.551         |240.75          |252.026         |286.132         |
|10s         |50         |0.0019     |52        |986.52          |1030.635        |1051.282        |1101.016        |

* The offline model_repo's overview:

![overview](./Overview.JPG)

### Online Model Perf
* Aishell2, FP16, onnx, [U2++ Conformer](http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/aishell2/20210618_u2pp_conformer_exp.tar.gz)
Our chunksize is 16 * 4 * 10 = 640 ms, so we should care about the perf of latency less than 640ms so that it can be a realtime application.

<table border='0' cellpadding='0' cellspacing='0'>
<tr id='r0'>
<td>Model</td>
<td>Num Left Chunks</td>
<td>Concurrency</td>
<td>Avg Latency (ms)</td>
<td>Latency P95 (ms)</td>
<td>Latency P99 (ms)</td>
<td>Throughput (infer/s)</td>
<td>Platform</td>
 </tr>
 <tr id='r1'>
<td rowspan='2' class='x21'>ctc prefix beam search</td>
<td rowspan='4' class='x21'>5</td>
<td>50</td>
<td>78</td>
<td>110</td>
<td>134</td>
<td>644</td>
<td rowspan='4' class='x21'>onnx</td>
 </tr>
 <tr id='r2'>
<td>100</td>
<td>124</td>
<td>155</td>
<td>192</td>
<td>805</td>
 </tr>
 <tr id='r5'>
<td rowspan='2' class='x21'>attention rescoring</td>
<td>50</td>
<td>99</td>
<td>136</td>
<td>164</td>
<td>504</td>
 </tr>
 <tr id='r6'>
<td>100</td>
<td>172</td>
<td>227</td>
<td>272</td>
<td>585</td>
 </tr>
</table>

### Improve Accuracy

#### Add Lanaguage Model
* Add language model: set `--lm_path` in the `convert_start_server.sh`. Notice the path of your language model is the path in docker.
* You may refer to `wenet/bin/recognize_onnx.py` to run inference locally. If you want to add language model locally, you may refer to [here](https://github.com/Slyne/ctc_decoder/blob/master/README.md#usage)

#### Dynamic Left Chunks
For online model, training with dynamic left chunk option on will help further improve the model accuracy.
Let's take a look at the below table.

**WenetSpeech**
chunksize = 16
|Scoring                | left-full | left-8 | left-4 | left-2 | left-1 | left-0 |
|-----------------------|-----------|--------|--------|--------|--------|--------|
| DEV |
|ctc greedy             | 9.16        | 9.19   |9.12      |9.05    |9.08      | 9.3    |
|ctc prefix beam          | 9.1          | 9.11     |9.04      |8.98       |9          |9.23    |
|attention rescoring      | 8.76      |    8.75     |8.69      |8.62       |8.61      |8.74    |
| Test Meeting |
| ctc greedy              |18.43        |18.47     |18.53      |18.81     |19.27      |20.37   |
| ctc prefix beam          |18.27        |18.29     |18.35      |18.65     |19.13      |20.22   |
| attention rescoring      |17.77        |17.81     |17.9      |18.19     |18.66      |19.68   |
| Test Net |
|ctc greedy                |10.91        |10.93     |10.98      |11.12     |11.29      |11.77   |
|ctc prefix beam          |10.86        |10.88     |10.93      |11.07     |11.24      |11.72   |
|attention rescoring      |10.13        |10.16     |10.2      |10.34     |10.51      |10.93   |

With dynamic left = False in training and chunk size=16 when inferencing, we take all the left chunks:
|Scoring            | DEV | Test Meeting | Test Net |
|-------------------|-----|--------------|----------|
|ctc greedy         | 9.32| 18.79        | 11.02    |
|ctc prefix beam    | 9.25| 18.62        | 10.96    |
|attention rescoring| 8.87| 18.11        | 10.22    |


### Warning
* We only tested mandarin models in `wenetspeech` and `aishell2` and haven't tested models built with English corpus.
* The bpe models currently are not supported.
* You need to add a VAD module to split your long audio into segments.
* Please pad your audio in order to leverage triton inference [dynamic batching](https://github.com/triton-inference-server/server/blob/main/docs/architecture.md#models-and-schedulers) feature. We suggest you to pad your audio to nearest length such as 2s, 5s, 10s seconds.
* We're still working on improving this solution. If you find any issue, please raise an issue.
* For streaming pipeline, we haven't implemented `endpoint` feature, which means you have to cut your audio manually.
* Please use `client.py` as a reference for building an intermediate server between the true clients (e.g., web application) and triton inference server. You may add the padding strategy (for offline model), buffering, vad, endpoint or other preprocessing (for streaming model) in this intermediate server.

More info can be found on Triton documents. For example, if you want to deploy your application on scale, you may refer to [triton k8s deploy](https://github.com/triton-inference-server/server/blob/main/deploy/gcp/README.md).


## Reference
* [Triton inference server](https://github.com/triton-inference-server/server) & [Client](https://github.com/triton-inference-server/client)

## Acknowledge
This part originates from NVIDIA CISI project. We also have TTS and NLP solutions deployed on triton inference server.
If you are interested, please contact us.

Thanks to @RiverLiu @Jiawei and @day9011 for great effort in testing.
