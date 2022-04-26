# WeNet on IPU
This project is a minimal modification to WeNet to allow it to run on Graphcore IPU, currently we support training on `wenetspeech`, `aishell`, `librispeech` dataset, and it is easy to make a little changes to run with other dataset.



## Environment setup

First, [download](https://downloads.graphcore.ai) and install the Poplar SDK following the instructions in the [Getting Started guide](https://docs.graphcore.ai/en/latest/) for your Graphcore IPU system. Make sure to source the enable.sh scripts for Poplar and PopART.

Then, create a virtual environment, install the required packages.

```
virtualenv venv -p python3.6
source venv/bin/activate
pip3 install -r requirements.txt
````

## How to run

```
cd examples/wenetspeech/s0/
bash run.sh
```

## performance


|model|dataset|test set|decoding method|GPU|IPU|
|---|---|---|---|---|---|
|conformer|wenetspeech|Test_Net|attention rescoring|9.7|9.68|
|conformer|aishell|test|attention rescoring|4.62|4.68|
|conformer|librispeech|test_other|attention rescoring|8.72|8.5|
