FROM ubuntu:latest
MAINTAINER <zhendong.peng@qq.com>
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git cmake wget build-essential
RUN git clone https://github.com/wenet-e2e/wenet.git /home/wenet
ARG model=20210618_u2pp_conformer_libtorch.tar.gz
RUN wget -P /home https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/aishell2/$model
RUN tar -xzf /home/$model -C /home
ARG src=/home/wenet/runtime/libtorch
RUN cmake -B $src/build -S $src -DCMAKE_BUILD_TYPE=Release -DGRAPH_TOOLS=ON && cmake --build $src/build
