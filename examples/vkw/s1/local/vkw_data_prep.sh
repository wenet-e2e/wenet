#!/bin/bash

current_dir=$(PWD)
stage=0
. ./path.sh || exit 1;
[ ! -z data/vkw ] && echo "wget vkw challenge data to this directory" && exit 0

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    cd $current_dir/data/vkw/data
    for x in dat_lgv dat_liv dat_stv; do
        [ ! -f ${x}.tar.gz ] && echo "no ${x}.tar.gz " && exit 0
        [ ! -z $x ] && mkdir ${x} && tar zxf ${x}.tar.gz -C ${x}
    done
    
    cd $current_dir/data/vkw/label
    for x in lab_lgv lab_liv lab_stv; do
        [ ! -f ${x}.tar.gz ] && echo "no ${x}.tar.gz " && exit 0
        [ ! -z $x ] && mkdir ${x} && tar zxf ${x}.tar.gz -C ${x}
    done  
    cd $current_dir
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    y=lgv
    z=train_20210525_vkw_${y}_3kh_org_mfcchires
    for x in finetune_5h finetune_50h dev_5h test_20h; do
        [ ! -f data/vkw/label/lab_${y}/${z}/${x}/wav_ori.scp ] && \
            mv data/vkw/label/lab_${y}/${z}/${x}/wav.scp data/vkw/label/lab_${y}/${z}/${x}/wav_ori.scp && \
           sed 's/ffmpeg\ -i\ /ffmpeg\ -i\ \/apdcephfs\/share_1157259\/users\/yougenyuan\/software\/wenet\/examples\/vkw\/s0\/data\/vkw\/data\/dat_lgv\//g' data/vkw/label/lab_${y}/${z}/${x}/wav_ori.scp > data/vkw/label/lab_${y}/${z}/${x}/wav.scp    
    done
    
    y=liv
    z=train_20210525_vkw_${y}_500h_org_mfcchires
    for x in finetune_5h finetune_50h dev_5h test_20h; do
        [ ! -f data/vkw/label/lab_${y}/${z}/${x}/wav_ori.scp ] && \
            mv data/vkw/label/lab_${y}/${z}/${x}/wav.scp data/vkw/label/lab_${y}/${z}/${x}/wav_ori.scp && \
           sed 's/ffmpeg\ -i\ /ffmpeg\ -i\ \/apdcephfs\/share_1157259\/users\/yougenyuan\/software\/wenet\/examples\/vkw\/s0\/data\/vkw\/data\/dat_liv\//g' data/vkw/label/lab_${y}/${z}/${x}/wav_ori.scp > data/vkw/label/lab_${y}/${z}/${x}/wav.scp    
    done
    
    y=stv
    z=train_20210525_vkw_${y}_500h_org_mfcchires
    for x in finetune_5h finetune_50h dev_5h test_20h; do
        [ ! -f data/vkw/label/lab_${y}/${z}/${x}/wav_ori.scp ] && \
            mv data/vkw/label/lab_${y}/${z}/${x}/wav.scp data/vkw/label/lab_${y}/${z}/${x}/wav_ori.scp && \
           sed 's/ffmpeg\ -i\ /ffmpeg\ -i\ \/apdcephfs\/share_1157259\/users\/yougenyuan\/software\/wenet\/examples\/vkw\/s0\/data\/vkw\/data\/dat_stv\//g' data/vkw/label/lab_${y}/${z}/${x}/wav_ori.scp > data/vkw/label/lab_${y}/${z}/${x}/wav.scp
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
   x=train_20210525_vkw_ddt_1kh_org_wuwFbank
   [ ! -f data/${x}/text ] && echo "vkw trainset is missing" && exit 0
fi

echo "$0: vkw  data preparation succeeded"
exit 0;
