FROM ubuntu:latest
MAINTAINER <zhendong.peng@mobvoi.com>
ENV DEBIAN_FRONTEND=noninteractive
RUN sed -i s@/archive.ubuntu.com/@/mirrors.tuna.tsinghua.edu.cn/@g /etc/apt/sources.list
RUN apt-get update && apt-get install -y git cmake wget build-essential
RUN git clone https://github.com/wenet-e2e/wenet.git /home/wenet
ARG model=20210618_u2pp_conformer_libtorch.tar.gz
RUN wget -P /home https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/aishell2/$model
RUN tar -xzf /home/$model -C /home
ARG build=/home/wenet/runtime/server/x86/build
RUN mkdir $build && cd $build && cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build . --config Release
