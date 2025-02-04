FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git cmake wget build-essential python-is-python3 python3-pip google-perftools
RUN pip install torch==2.3.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install intel_extension_for_pytorch==2.3.0 pyyaml six intel-openmp
RUN ln -s /usr/lib/x86_64-linux-gnu/libtcmalloc.so.4 /usr/lib/x86_64-linux-gnu/libtcmalloc.so

RUN git clone https://github.com/wenet-e2e/wenet.git /home/wenet
ENV OMP_NUM_THREADS=1
ARG src=/home/wenet/runtime/ipex
RUN cmake -B $src/build -S $src -DCMAKE_BUILD_TYPE=Release && cmake --build $src/build -j32
ENV LD_LIBRARY_PATH=$src/fc_base/libtorch-src/lib/:$LD_LIBRARY_PATH
WORKDIR /home/wenet/
