# Install

## Install prebuilt python package

If you just want to use WeNet as a python package for speech recognition application,
just install it by `pip`, please note python 3.6+ is required.
``` sh
pip3 install wenetruntime
```

And please see [doc](https://github.com/wenet-e2e/wenet/tree/main/runtime/binding/python) for usage.


## Install for training

- Clone the repo
``` sh
git clone https://github.com/wenet-e2e/wenet.git
```

- Install Conda: please see https://docs.conda.io/en/latest/miniconda.html
- Create Conda env:

``` sh
conda create -n wenet python=3.8
conda activate wenet
pip install -r requirements.txt
```

## Build for deployment

Optionally, if you want to use x86 runtime or language model(LM),
you have to build the runtime as follows. Otherwise, you can just ignore this step.

``` sh
# runtime build requires cmake 3.14 or above
cd runtime/libtorch
mkdir build && cd build && cmake -DGRAPH_TOOLS=ON .. && cmake --build .
```

Please see [doc](https://github.com/wenet-e2e/wenet/tree/main/runtime) for building
runtime on more platforms and OS.
