# WeNet on IPU
This is the repo we made minimized changes on the WeNet to allow it running on Graphcore IPU, currently we support training on `wenetspeech` dataset, and it is easy to make a little changes to run with other dataset. 



## Environment setup

First, [download](https://downloads.graphcore.ai) and install the Poplar SDK following the instructions in the [Getting Started guide](https://docs.graphcore.ai/en/latest/) for your Graphcore IPU system. Make sure to source the enable.sh scripts for Poplar and PopART.

Then, create a virtual environment, install the required packages.

```
virtualenv venv -p python3.6
source venv/bin/activate
pip3 install -r requirements.txt
````

