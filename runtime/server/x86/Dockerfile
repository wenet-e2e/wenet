FROM ubuntu:latest
MAINTAINER <zhendong.peng@mobvoi.com>
ENV DEBIAN_FRONTEND=noninteractive
RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list
RUN apt-get update && apt-get install -y git cmake wget build-essential
RUN git clone https://github.com/mobvoi/wenet.git /home/wenet
ARG model=20210327_unified_transformer_exp_server.tar.gz
RUN wget -P /home http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/aishell2/$model
RUN tar -xzf /home/$model -C /home
ARG build=/home/wenet/runtime/server/x86/build
RUN mkdir $build && cmake -S $build/.. -B $build

