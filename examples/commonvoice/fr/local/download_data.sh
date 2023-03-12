#!/usr/bin/env bash
if [ $# -le 1 ]; then
    echo "Args_Error:Two parameters are required."
    exit 1;
fi
download_path=$1
data_France=$2
wget -O ${download_path}/tmp.zip https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/cv-corpus-8.0-2022-01-19/cv-corpus-8.0-2022-01-19-fr.tar.gz
tar -xvf ${download_path}/tmp.zip  -C ${data_France}
rm -rf ${download_path}/tmp.zip