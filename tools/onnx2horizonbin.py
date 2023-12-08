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
"""NOTE(xcsong): Currently, we only support
1. specific conformer encoder architecture, see:
    encoder: conformer
    encoder_conf:
      activation_type: **must be** relu
      attention_heads: 2 or 4 or 8 or any number divisible by output_size
      causal: **must be** true
      cnn_module_kernel: 1 ~ 7
      cnn_module_norm: **must be** batch_norm
      input_layer: **must be** conv2d8
      linear_units: 1 ~ 2048
      normalize_before: **must be** true
      num_blocks: 1 ~ 12
      output_size: 1 ~ 512
      pos_enc_layer_type: **must be** no_pos
      selfattention_layer_type: **must be** selfattn
      use_cnn_module: **must be** true
      use_dynamic_chunk: **must be** true
      use_dynamic_left_chunk: **must be** true

2. specific decoding method: ctc_greedy_search
"""

import argparse
import copy
import logging
import os
import sys
import random
import torch
import yaml
import numpy as np

from torch.utils.data import DataLoader

from wenet.utils.common import remove_duplicates_and_blank
from wenet.dataset.dataset import Dataset
from wenet.utils.checkpoint import load_checkpoint
from wenet.utils.init_model import init_model
from wenet.utils.init_tokenizer import init_tokenizer
from wenet.bin.export_onnx_cpu import to_numpy
from wenet.bin.export_onnx_bpu import export_encoder, export_ctc

try:
    import hbdk  # noqa: F401
    import horizon_nn  # noqa: F401
    from horizon_tc_ui import HB_ONNXRuntime
except ImportError:
    print('Please install hbdk,horizon_nn,horizon_tc_ui !')
    sys.exit(1)

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def save_data(tensor, dirs, prefix):
    if tensor.requires_grad:
        data = tensor.detach().numpy().astype(np.float32)
    else:
        data = tensor.numpy().astype(np.float32)
    os.makedirs(dirs, exist_ok=True)
    data.tofile(dirs + "/" + prefix + ".bin")


def make_calibration_data(enc, args, conf, tokenizer):
    conf['shuffle'] = True
    logger.info(conf)
    dataset = Dataset("shard",
                      args.cali_datalist,
                      tokenizer,
                      conf,
                      partition=False)
    dataloader = DataLoader(dataset, batch_size=None, num_workers=0)

    subsampling = enc.embed.subsampling_rate
    context = enc.embed.right_context + 1  # Add current frame
    stride = subsampling * args.chunk_size
    decoding_window = (args.chunk_size - 1) * subsampling + context
    required_cache_size = args.chunk_size * args.num_decoding_left_chunks
    num_layers = len(enc.encoders)
    head, d_k = enc.encoders[0].self_attn.h, enc.encoders[0].self_attn.d_k
    dim, lorder = enc._output_size, enc.encoders[0].conv_module.lorder
    chunk_size, left_chunks = args.chunk_size, args.num_decoding_left_chunks
    cal_data_dir = os.path.join(args.output_dir, 'cal_data_dir')
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= args.max_samples:
            break
        if batch_idx % 100 == 0:
            logger.info("processed {} samples.".format(batch_idx))
        keys, feats, target, feats_lengths, target_lengths = batch
        num_frames, prefix = feats.size(1), keys[0]
        att_cache = torch.zeros(
            [1, head * num_layers, d_k * 2, required_cache_size],
            dtype=feats.dtype,
            device=feats.device)
        att_mask = torch.ones(
            [1, head, chunk_size, required_cache_size + chunk_size],
            dtype=feats.dtype,
            device=feats.device)
        att_mask[:, :, :, :required_cache_size] = 0
        cnn_cache = torch.zeros([1, dim, num_layers, lorder],
                                dtype=feats.dtype,
                                device=feats.device)

        # Feed forward overlap input step by step
        random_high = (num_frames - context) // stride
        num_rand = random.randint(0, random_high)
        for i, cur in enumerate(range(0, num_frames - context + 1, stride)):
            att_mask[:, :, :, -(chunk_size * (i + 1)):] = 1
            end = min(cur + decoding_window, num_frames)
            chunk = feats[:, cur:end, :].unsqueeze(0)  # (1, 1, window, mel)
            if end == num_frames and end - cur < decoding_window:  # last chunk
                pad_len = decoding_window - (end - cur)  # 67 - (35)
                pad_chunk = torch.zeros((1, 1, pad_len, chunk.size(-1)),
                                        device=feats.device)
                chunk = torch.cat((chunk, pad_chunk),
                                  dim=2)  # (1, 1, win, mel)
                if pad_len >= subsampling:
                    att_mask[:, :, :, -(pad_len // subsampling):] = 0
            if i == num_rand:
                save_data(chunk, "{}/chunk".format(cal_data_dir),
                          prefix + "." + str(i))
                save_data(att_cache, "{}/att_cache".format(cal_data_dir),
                          prefix + "." + str(i))
                save_data(cnn_cache, "{}/cnn_cache".format(cal_data_dir),
                          prefix + "." + str(i))
                save_data(att_mask, "{}/att_mask".format(cal_data_dir),
                          prefix + "." + str(i))
            (y, att_cache, cnn_cache) = enc.forward(xs=chunk,
                                                    att_cache=att_cache,
                                                    cnn_cache=cnn_cache,
                                                    att_mask=att_mask)
            # NOTE(xcsong): It's fast to calibrate ctc.onnx,
            #   so it's okay to save all chunks
            save_data(y, "{}/hidden".format(cal_data_dir),
                      prefix + "." + str(i))


def check_wer(enc, ctc, args, conf, tokenizer):
    conf['shuffle'] = False
    dataset = Dataset("shard",
                      args.wer_datalist,
                      tokenizer,
                      conf,
                      partition=False)
    dataloader = DataLoader(dataset, batch_size=None, num_workers=0)
    char_dict = {v: k for k, v in tokenizer.symbol_table.items()}
    eos = len(char_dict) - 1

    enc_session = HB_ONNXRuntime(
        model_file=args.output_dir +
        "/hb_makertbin_output_encoder/encoder_quantized_model.onnx")
    ctc_session = HB_ONNXRuntime(
        model_file=args.output_dir +
        "/hb_makertbin_output_ctc/ctc_quantized_model.onnx")
    torch_file = open(args.output_dir + "/torch_text", 'w', encoding="utf-8")
    onnx_file = open(args.output_dir + "/onnx_text", 'w', encoding="utf-8")
    subsampling = enc.embed.subsampling_rate
    context = enc.embed.right_context + 1  # Add current frame
    stride = subsampling * args.chunk_size
    decoding_window = (args.chunk_size - 1) * subsampling + context
    required_cache_size = args.chunk_size * args.num_decoding_left_chunks
    num_layers = len(enc.encoders)
    head, d_k = enc.encoders[0].self_attn.h, enc.encoders[0].self_attn.d_k
    dim, lorder = enc._output_size, enc.encoders[0].conv_module.lorder
    chunk_size, left_chunks = args.chunk_size, args.num_decoding_left_chunks
    for batch_idx, batch in enumerate(dataloader):
        keys, feats, target, feats_lengths, target_lengths = batch
        num_frames, prefix = feats.size(1), keys[0]
        att_cache = torch.zeros(
            [1, head * num_layers, d_k * 2, required_cache_size],
            dtype=feats.dtype,
            device=feats.device)
        att_mask = torch.ones(
            [1, head, chunk_size, required_cache_size + chunk_size],
            dtype=feats.dtype,
            device=feats.device)
        att_mask[:, :, :, :required_cache_size] = 0
        cnn_cache = torch.zeros([1, dim, num_layers, lorder],
                                dtype=feats.dtype,
                                device=feats.device)
        onnx_att_cache = to_numpy(att_cache)
        onnx_cnn_cache = to_numpy(cnn_cache)

        # Feed forward overlap input step by step
        torch_out, onnx_out = [], []
        for i, cur in enumerate(range(0, num_frames - context + 1, stride)):
            att_mask[:, :, :, -(chunk_size * (i + 1)):] = 1
            end = min(cur + decoding_window, num_frames)
            chunk = feats[:, cur:end, :].unsqueeze(0)  # (1, 1, window, mel)
            if end == num_frames and end - cur < decoding_window:  # last chunk
                pad_len = decoding_window - (end - cur)  # 67 - (35)
                pad_chunk = torch.zeros((1, 1, pad_len, chunk.size(-1)),
                                        device=feats.device)
                chunk = torch.cat((chunk, pad_chunk),
                                  dim=2)  # (1, 1, win, mel)
                if pad_len >= subsampling:
                    att_mask[:, :, :, -(pad_len // subsampling):] = 0
            # Torch model
            (y, att_cache, cnn_cache) = enc.forward(xs=chunk,
                                                    att_cache=att_cache,
                                                    cnn_cache=cnn_cache,
                                                    att_mask=att_mask)
            torch_out.append(ctc.forward(y).transpose(1, 3).squeeze(2))
            # Quantized onnx model
            ort_inputs = {
                'chunk': to_numpy(chunk),
                'att_cache': onnx_att_cache,
                'cnn_cache': onnx_cnn_cache,
                'att_mask': to_numpy(att_mask)
            }
            ort_outs = enc_session.run_feature(enc_session.output_names,
                                               ort_inputs,
                                               input_offset=0)
            onnx_att_cache, onnx_cnn_cache = ort_outs[1], ort_outs[2]
            onnx_y = ctc_session.run_feature(ctc_session.output_names,
                                             {'hidden': ort_outs[0]},
                                             input_offset=0)
            onnx_out.append(
                torch.from_numpy(
                    np.squeeze(onnx_y[0].transpose(0, 3, 2, 1), axis=2)))

        def post_process(list_out, file_obj, keys):
            probs = torch.cat(list_out, dim=1)
            maxlen = probs.size(1)
            topk_prob, topk_index = probs.topk(1, dim=2)  # (B, maxlen, 1)
            topk_index = topk_index.view(1, maxlen)  # (B, maxlen)
            hyps = [hyp.tolist() for hyp in topk_index]
            scores = topk_prob.max(1)
            hyps = [remove_duplicates_and_blank(hyp) for hyp in hyps]
            for i, key in enumerate(keys):
                content = ''
                for w in hyps[i]:
                    if w == eos:
                        break
                    content += char_dict[w]
                file_obj.write('{} {}\n'.format(key, content))
            return key, content

        if len(torch_out) > 0 and len(onnx_out) > 0:
            key, content = post_process(torch_out, torch_file, keys)
            logger.info('torch: {} {}'.format(key, content))
            key, content = post_process(onnx_out, onnx_file, keys)
            logger.info('onnx : {} {}'.format(key, content))
    torch_file.close()
    onnx_file.close()


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
  # input_batch: 1
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
  calibration_type: '{}'
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
        enc_onnx_path, "encoder", enc_log_path, enc_dic['input_name'],
        enc_dic['input_type'], enc_dic['input_layout_train'],
        enc_dic['input_shape'], enc_dic['norm_type'], enc_dic['input_type'],
        enc_dic['input_layout_rt'], enc_cal_data, args.calibration_type,
        args.extra_ops_run_on_cpu, "")
    ctc_config = template.format(
        ctc_onnx_path, "ctc", ctc_log_path, ctc_dic['input_name'],
        ctc_dic['input_type'], ctc_dic['input_layout_train'],
        ctc_dic['input_shape'], ctc_dic['norm_type'], ctc_dic['input_type'],
        ctc_dic['input_layout_rt'], ctc_cal_data, "default", "", "")
    with open(output_dir + "/config_encoder.yaml", "w") as enc_yaml:
        enc_yaml.write(enc_config)
    with open(output_dir + "/config_ctc.yaml", "w") as ctc_yaml:
        ctc_yaml.write(ctc_config)


def get_args():
    parser = argparse.ArgumentParser(
        description='convert onnx to horizon .bin')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--output_dir', required=True, help='output directory')
    parser.add_argument('--chunk_size',
                        required=True,
                        type=int,
                        help='decoding chunk size')
    parser.add_argument('--num_decoding_left_chunks',
                        required=True,
                        type=int,
                        help='cache chunks')
    parser.add_argument('--reverse_weight',
                        default=0.5,
                        type=float,
                        help='reverse_weight in attention_rescoing')
    parser.add_argument('--max_samples',
                        type=int,
                        required=True,
                        help='maximum samples')
    parser.add_argument('--cali_datalist',
                        type=str,
                        default=None,
                        help='make calibration data')
    parser.add_argument('--wer_datalist',
                        type=str,
                        default=None,
                        help='check wer')
    parser.add_argument('--wer_text', type=str, default=None, help='check wer')
    parser.add_argument('--ln_run_on_bpu',
                        action='store_true',
                        help='layernorm running on bpu')
    parser.add_argument('--extra_ops_run_on_cpu',
                        type=str,
                        default=None,
                        help='extra operations running on cpu.')
    parser.add_argument('--calibration_type',
                        type=str,
                        default='default',
                        help='kl / max / default.')
    return parser


if __name__ == '__main__':
    random.seed(777)
    parser = get_args()
    args = parser.parse_args()
    # NOTE(xcsong): X3 BPU only support static shapes
    assert args.chunk_size > 0
    assert args.num_decoding_left_chunks > 0
    os.system("mkdir -p " + args.output_dir)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    model = init_model(args, configs)
    load_checkpoint(model, args.checkpoint)
    tokenizer = init_tokenizer(configs)
    model.eval()

    args.feature_size = configs['input_dim']
    args.output_size = model.encoder.output_size()
    args.decoding_window = (args.chunk_size - 1) * \
        model.encoder.embed.subsampling_rate + \
        model.encoder.embed.right_context + 1

    logger.info("Stage-1: Export onnx")
    enc, enc_session = export_encoder(model, args)
    ctc, ctc_session = export_ctc(model, args)

    conf = copy.deepcopy(configs['dataset_conf'])
    conf['filter_conf']['max_length'] = 102400
    conf['filter_conf']['min_length'] = 0
    conf['filter_conf']['token_max_length'] = 102400
    conf['filter_conf']['token_min_length'] = 0
    conf['filter_conf']['max_output_input_ratio'] = 102400
    conf['filter_conf']['min_output_input_ratio'] = 0
    conf['speed_perturb'] = False
    conf['spec_aug'] = False
    conf['spec_sub'] = False
    conf['spec_trim'] = False
    conf['shuffle'] = False
    conf['sort'] = False
    if 'fbank_conf' in conf:
        conf['fbank_conf']['dither'] = 0.0
    elif 'mfcc_conf' in conf:
        conf['mfcc_conf']['dither'] = 0.0
    conf['batch_conf']['batch_type'] = "static"
    conf['batch_conf']['batch_size'] = 1

    if args.cali_datalist is not None:
        logger.info("Stage-2: Generate config")
        # FIXME(xcsong): Remove hard code
        logger.info("torch version: {}".format(torch.__version__))
        if int(torch.__version__[:4].split('.')[1]) >= 13:
            args.extra_ops_run_on_cpu = "/Split;" + \
                "/encoders.0/self_attn/Split;/encoders.1/self_attn/Split;" + \
                "/encoders.2/self_attn/Split;/encoders.3/self_attn/Split;" + \
                "/encoders.4/self_attn/Split;/encoders.5/self_attn/Split;" + \
                "/encoders.6/self_attn/Split;/encoders.7/self_attn/Split;" + \
                "/encoders.8/self_attn/Split;/encoders.9/self_attn/Split;" + \
                "/encoders.10/self_attn/Split;/encoders.11/self_attn/Split;" + \
                "/encoders.0/self_attn/Mul;/encoders.1/self_attn/Mul;" + \
                "/encoders.2/self_attn/Mul;/encoders.3/self_attn/Mul;" + \
                "/encoders.4/self_attn/Mul;/encoders.5/self_attn/Mul;" + \
                "/encoders.6/self_attn/Mul;/encoders.7/self_attn/Mul;" + \
                "/encoders.8/self_attn/Mul;/encoders.9/self_attn/Mul;" + \
                "/encoders.10/self_attn/Mul;/encoders.11/self_attn/Mul;"
        else:
            args.extra_ops_run_on_cpu = "Split_17;Split_67;Split_209;" + \
                "Split_351;Split_493;Split_635;Split_777;Split_919;Split_1061;" + \
                "Split_1203;Split_1345;Split_1487;Split_1629;" + \
                "Mul_72;Mul_214;Mul_356;Mul_498;Mul_640;Mul_782;" + \
                "Mul_924;Mul_1066;Mul_1208;Mul_1350;Mul_1492;Mul_1634;"
        generate_config(enc_session, ctc_session, args)

        logger.info("Stage-3: Make calibration data")
        make_calibration_data(enc, args, conf, tokenizer)

        output_dir = os.path.realpath(args.output_dir)
        logger.info("Stage-4: Make ctc.bin")
        os.system("cd {} && mkdir -p hb_makertbin_log_ctc".format(output_dir) +
                  " && cd hb_makertbin_log_ctc &&" +
                  " hb_mapper makertbin --model-type \"onnx\" --config \"{}\"".
                  format(output_dir + "/config_ctc.yaml"))
        logger.info("Stage-5: Make encoder.bin")
        os.system(
            "cd {} && mkdir -p hb_makertbin_log_encoder ".format(output_dir) +
            " && cd hb_makertbin_log_encoder &&" +
            " hb_mapper makertbin --model-type \"onnx\" --config \"{}\"".
            format(output_dir + "/config_encoder.yaml"))

    if args.wer_datalist is not None:
        logger.info(
            "Stage-6: Check wer between torch model and quantized onnx")
        assert args.wer_text is not None
        check_wer(enc, ctc, args, conf, tokenizer)
        os.system(
            "python3 tools/compute-wer.py --char=1 --v=1 {} {} > {}".format(
                args.wer_text, args.output_dir + "/torch_text",
                args.output_dir + "/torch_wer"))
        os.system(
            "python3 tools/compute-wer.py --char=1 --v=1 {} {} > {}".format(
                args.wer_text, args.output_dir + "/onnx_text",
                args.output_dir + "/onnx_wer"))
        os.system("tail {} {}".format(args.output_dir + "/torch_wer",
                                      args.output_dir + "/onnx_wer"))
