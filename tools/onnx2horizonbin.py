# Copyright (c) 2022, Horizon Inc. Xingchen Song (sxc19@tsinghua.org.cn)
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

import logging
import os
import sys
import yaml

from wenet.bin.export_onnx_cpu import get_args
from wenet.bin.export_onnx_bpu import export_encoder, export_ctc
from wenet.utils.checkpoint import load_checkpoint
from wenet.utils.init_model import init_model


try:
    import hbdk  # noqa: F401
    import horizon_nn  # noqa: F401
    import horizon_tc_ui  # noqa: F401
except ImportError:
    print('Please install hbdk,horizon_nn,horizon_tc_ui !')
    sys.exit(1)


logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def make_calibration_data(enc, args):
    pass


def generate_config(enc_session, ctc_session, args):
    template = """
# 模型参数组
model_parameters:
  # 原始Onnx浮点模型文件
  onnx_model: '{}'
  # 转换的目标AI芯片架构
  march: 'bernoulli2'
  # 模型转换输出的用于上板执行的模型文件的名称前缀
  output_model_file_prefix: '{}'
  # 模型转换输出的结果的存放目录
  working_dir: '{}'
  # 指定转换后混合异构模型是否保留输出各层的中间结果的能力
  layer_out_dump: False
  # 转换过程中日志生成级别
  log_level: 'debug'
# 输入信息参数组
input_parameters:
  # 原始浮点模型的输入节点名称
  input_name: '{}'
  # 原始浮点模型的输入数据格式（数量/顺序与input_name一致）
  input_type_train: '{}'
  # 原始浮点模型的输入数据排布（数量/顺序与input_name一致）
  input_layout_train: '{}'
  # 原始浮点模型的输入数据尺寸
  input_shape: '{}'
  # 网络实际执行时，输入给网络的batch_size  默认值为1
  input_batch: 1
  # 在模型中添加的输入数据预处理方法
  norm_type: '{}'
  # 预处理方法的图像减去的均值; 如果是通道均值，value之间必须用空格分隔
  # mean_value: ''
  # 预处理方法的图像缩放比例，如果是通道缩放比例，value之间必须用空格分隔
  # scale_value: ''
  # 转换后混合异构模型需要适配的输入数据格式（数量/顺序与input_name一致）
  input_type_rt: '{}'
  # 输入数据格式的特殊制式
  input_space_and_range: ''
  # 转换后混合异构模型需要适配的输入数据排布（数量/顺序与input_name一致）
  input_layout_rt: '{}'
# 校准参数组
calibration_parameters:
  # 模型校准使用的标定样本的存放目录
  cal_data_dir: '{}'
  # 开启图片校准样本自动处理（skimage read resize到输入节点尺寸）
  preprocess_on: False
  # 校准使用的算法类型
  calibration_type: 'default'
  # max 校准方式的参数
  max_percentile: 1.0
  # 强制指定OP在CPU上运行
  run_on_cpu: '{}'
  # 强制指定OP在BPU上运行
  run_on_bpu: '{}'
# 编译参数组
compiler_parameters:
  # 编译策略选择
  compile_mode: 'latency'
  # 是否打开编译的debug信息
  debug: False
  # 模型运行核心数
  core_num: 1
  # 模型编译的优化等级选择
  optimize_level: 'O3'
"""
    output_dir = os.path.realpath(args.output_dir)
    cal_data_dir = os.path.join(output_dir, 'cal_data_dir')
    os.makedirs(cal_data_dir, exist_ok=True)
    enc_dic = enc_session.get_modelmeta().custom_metadata_map
    enc_onnx_path = os.path.join(output_dir, 'encoder.onnx')
    enc_log_path = os.path.join(output_dir, 'hb_makertbin_output_encoder')
    enc_cal_data = ";".join(
        [cal_data_dir + "/" + x for x in enc_dic['input_name'].split(';')])
    ctc_dic = ctc_session.get_modelmeta().custom_metadata_map
    ctc_onnx_path = os.path.join(output_dir, 'ctc.onnx')
    ctc_log_path = os.path.join(output_dir, 'hb_makertbin_output_ctc')
    ctc_cal_data = ";".join(
        [cal_data_dir + "/" + x for x in ctc_dic['input_name'].split(';')])
    enc_config = template.format(
        enc_onnx_path, "encoder", enc_log_path,
        enc_dic['input_name'], enc_dic['input_type'],
        enc_dic['input_layout_train'], enc_dic['input_shape'],
        enc_dic['norm_type'], enc_dic['input_type'], enc_dic['input_layout_rt'],
        enc_cal_data, "", "")
    ctc_config = template.format(
        ctc_onnx_path, "ctc", ctc_log_path,
        ctc_dic['input_name'], ctc_dic['input_type'],
        ctc_dic['input_layout_train'], ctc_dic['input_shape'],
        ctc_dic['norm_type'], ctc_dic['input_type'], ctc_dic['input_layout_rt'],
        ctc_cal_data, "", "")
    with open(output_dir + "/config_encoder.yaml", "w") as enc_yaml:
        enc_yaml.write(enc_config)
    with open(output_dir + "/config_ctc.yaml", "w") as ctc_yaml:
        ctc_yaml.write(ctc_config)


if __name__ == '__main__':
    args = get_args()
    args.ln_run_on_bpu = True
    args.attn_run_on_bpu = True
    # NOTE(xcsong): X3 BPU only support static shapes
    assert args.chunk_size > 0
    assert args.num_decoding_left_chunks > 0
    os.system("mkdir -p " + args.output_dir)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    model = init_model(configs)
    load_checkpoint(model, args.checkpoint)
    model.eval()

    args.feature_size = configs['input_dim']
    args.output_size = model.encoder.output_size()
    args.decoding_window = (args.chunk_size - 1) * \
        model.encoder.embed.subsampling_rate + \
        model.encoder.embed.right_context + 1

    enc, enc_session = export_encoder(model, args)
    ctc, ctc_session = export_ctc(model, args)

    generate_config(enc_session, ctc_session, args)
    make_calibration_data(enc, args)
