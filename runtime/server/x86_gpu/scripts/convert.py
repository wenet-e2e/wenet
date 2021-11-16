#!/usr/bin/env python3
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import yaml
import os
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate config.pbtxt for model_repo')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--vocab', required=True, help='vocabulary file, words.txt')
    parser.add_argument('--model_repo', required=True, help='model repo directory')
    parser.add_argument('--onnx_model_dir', default=True, type=str, required=False,
                        help="onnx model path")
    parser.add_argument('--lm_path', default=None, type=str, required=False,
                        help="the additional language model path")
    args = parser.parse_args()
    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    with open(os.path.join(args.onnx_model_dir, 'config.yaml'), 'r') as fin:
        onnx_configs = yaml.load(fin, Loader=yaml.FullLoader)

    model_params = {
        "#beam_size": 10,
        "#num_mel_bins": 80,
        "#frame_shift": 10,
        "#frame_length": 25,
        "#sample_rate": 16000,
        "#encoder_out_feat_size": 256,
        "#lm_path": "",
        "#bidecoder": 0,
        "#vocabulary_path": "",
        "#vocab_size": 0,
        "#DTYPE": "FP32"
    }

    # fill values
    model_params["#beam_size"] = onnx_configs["beam_size"]
    if onnx_configs["fp16"]:
        model_params["#DTYPE"] = "FP16"
    feature_conf = configs["dataset_conf"]["fbank_conf"]
    model_params["#num_mel_bins"] = feature_conf["num_mel_bins"]
    model_params["#frame_shift"] = feature_conf["frame_shift"]
    model_params["#frame_length"] = feature_conf["frame_length"]
    dataset_conf = configs["dataset_conf"]["resample_conf"]
    model_params["#sample_rate"] = dataset_conf["resample_rate"]
    model_params["#encoder_out_feat_size"] = configs["encoder_conf"]["output_size"]
    model_params["#lm_path"] = args.lm_path
    if configs["decoder"].startswith("bi"):
        model_params["#bidecoder"] = 1
    model_params["#vocabulary_path"] = args.vocab
    model_params["#vocab_size"] = configs["output_dim"]

    for model in os.listdir(args.model_repo):
        template = "config_template.pbtxt"
        if "decoder" == model and model_params["#bidecoder"] == 0:
            template = "config_template2.pbtxt"

        model_dir = os.path.join(args.model_repo, model)
        out = os.path.join(model_dir, "config.pbtxt")
        out = open(out, "w")

        with open(os.path.join(model_dir, template), "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                for key, value in model_params.items():
                    # currently, the exported encoder output size is -1
                    # means this output dim is dynamic
                    # we are checking this issue
                    if key == "#encoder_out_feat_size" and \
                       model == "encoder":
                        value = "-1"
                    line = line.replace(key, str(value))
                out.write(line)
        out.close()

        if model in ("decoder", "encoder"):
            if onnx_configs["fp16"]:
                model_name = model + "_fp16.onnx"
            else:
                model_name = model + ".onnx"
            source_model = os.path.join(args.onnx_model_dir, model_name)
            target_model = os.path.join(model_dir, "1", model + ".onnx")
            res = subprocess.call(["cp", source_model, target_model], shell=False)
