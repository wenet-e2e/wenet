import argparse
import os
import torch

import yaml


def convert_to_wenet_yaml(wenet_yaml_path: str):
    configs = {}
    configs['input_dim'] = 80
    # whisper token nums
    configs['output_dim'] = 51866

    configs = {}
    configs['input_dim'] = 80
    # whisper token nums
    configs['output_dim'] = 1024

    configs['encoder'] = 'conformer'
    configs['encoder_conf'] = {}
    configs['encoder_conf']['causal'] = True
    configs['encoder_conf']['gradient_checkpointing'] = True
    configs['encoder_conf']['input_layer'] = 'stack_n_frames'
    configs['encoder_conf']['output_size'] = 1024
    configs['encoder_conf']['attention_heads'] = 16
    configs['encoder_conf']['linear_units'] = 4096
    configs['encoder_conf']['num_blocks'] = 24
    configs['encoder_conf']['dropout_rate'] = 0.1
    configs['encoder_conf']['positional_dropout_rate'] = 0.0
    configs['encoder_conf']['attention_dropout_rate'] = 0.0
    configs['encoder_conf']['normalize_before'] = True
    configs['encoder_conf']['use_dynamic_chunk'] = False
    configs['encoder_conf']['use_dynamic_left_chunk'] = False
    configs['encoder_conf']['pos_enc_layer_type'] = "no_pos"
    configs['encoder_conf']['static_chunk_size'] = -1
    configs['encoder_conf']['activation_type'] = "swish"
    configs['encoder_conf']['conv_bias'] = False
    configs['encoder_conf']['selfattention_layer_type'] = 'shaw_rel_selfattn'
    configs['encoder_conf']['cnn_module_kernel'] = 31
    configs['encoder_conf']['cnn_module_norm'] = 'layer_norm'

    # dummy decoder
    # TODO(Mddct): To use whisper's decoder here
    configs['decoder'] = 'transformer'
    configs['decoder_conf'] = {}
    configs['decoder_conf']['attention_head'] = 16
    configs['decoder_conf']['linear_units'] = 4096
    configs['decoder_conf']['num_blocks'] = 6
    configs['decoder_conf']['dropout_rate'] = 0.1
    configs['decoder_conf']['positional_dropout_rate'] = 0.1
    configs['decoder_conf']['self_attention_dropout_rate'] = 0.0
    configs['decoder_conf']['src_attention_dropout_rate'] = 0.0

    configs['cmvn'] = None
    configs['cmvn_conf'] = {}
    configs['cmvn_conf']['cmvn_file'] = None
    configs['cmvn_conf']['is_json_cmvn'] = None

    configs['model'] = "asr_model"
    configs['model_conf'] = {}
    configs['model_conf']['ctc_weight'] = 0.3
    configs['model_conf']['lsm_weight'] = 0.1
    configs['model_conf']['length_normalized_loss'] = False

    configs['dataset'] = "asr"
    configs['dataset_conf'] = {}
    configs['dataset_conf']['filter_conf'] = {}
    configs['dataset_conf']['filter_conf'][
        'max_length'] = 419000  # 1/2 subsample # noqa
    configs['dataset_conf']['filter_conf']['min_length'] = 0
    configs['dataset_conf']['filter_conf']['token_max_length'] = 400
    configs['dataset_conf']['filter_conf']['token_min_length'] = 1
    configs['dataset_conf']['resample_conf'] = {}
    configs['dataset_conf']['resample_conf']['resample_rate'] = 16000
    configs['dataset_conf']['speed_perturb'] = False
    configs['dataset_conf']['spec_aug'] = True
    configs['dataset_conf']['spec_aug_conf'] = {}
    configs['dataset_conf']['spec_aug_conf']['num_t_mask'] = 2
    configs['dataset_conf']['spec_aug_conf']['num_f_mask'] = 2
    configs['dataset_conf']['spec_aug_conf']['max_t'] = 50
    configs['dataset_conf']['spec_aug_conf']['max_f'] = 10
    configs['dataset_conf']['spec_sub'] = True
    configs['dataset_conf']['spec_sub_conf'] = {}
    configs['dataset_conf']['spec_sub_conf']['num_t_sub'] = 3
    configs['dataset_conf']['spec_sub_conf']['max_t'] = 30
    configs['dataset_conf']['spec_trim'] = False
    configs['dataset_conf']['shuffle'] = True
    configs['dataset_conf']['shuffle_conf'] = {}
    configs['dataset_conf']['shuffle_conf']['shuffle_size'] = 1500
    configs['dataset_conf']['sort'] = True
    configs['dataset_conf']['sort_conf'] = {}
    configs['dataset_conf']['sort_conf']['sort_size'] = 500
    configs['dataset_conf']['feats_type'] = "fbank"
    configs['dataset_conf']['batch_conf'] = {}
    configs['dataset_conf']['batch_conf']['batch_type'] = 'dynamic'
    configs['dataset_conf']['batch_conf']['batch_size'] = 26
    configs['dataset_conf']['batch_conf']['max_frames_in_batch'] = 12000

    configs['grad_clip'] = 5
    configs['accum_grad'] = 4
    configs['max_epoch'] = 100
    configs['log_interval'] = 100

    configs['optim'] = "adam"
    configs['optim_conf'] = {}
    configs['optim_conf']['lr'] = 0.0005
    configs['scheduler'] = "warmuplr"
    configs['scheduler_conf'] = {}
    configs['scheduler_conf']['warmup_steps'] = 12000

    with open(wenet_yaml_path, '+w') as f:
        f.write(yaml.dump(configs))
        f.flush()

    print(configs)


def convert_to_wenet_state_dict(w2vbert_conformer_state_dict,
                                wenet_state_dict_path):

    wenet_state_dict = {}
    print("==============start CKPT Conversion =========================")
    conformer_state_dict = w2vbert_conformer_state_dict
    wenet_state_dict = {}
    for name in conformer_state_dict.keys():
        old_name = name
        name = name.replace('encoder.layers', 'encoder.encoders')
        name = name.replace("ffn1_layer_norm", "norm_ff_macaron")
        name = name.replace("self_attn_layer_norm", "norm_mha")
        name = name.replace("conv_layer_norm", "norm_conv")
        name = name.replace("ffn2_layer_norm", "norm_ff")
        name = name.replace("self_attn.q_proj", "self_attn.linear_q")
        name = name.replace("self_attn.k_proj", "self_attn.linear_k")
        name = name.replace("self_attn.v_proj", "self_attn.linear_v")
        name = name.replace("self_attn.output_proj", "self_attn.linear_out")
        name = name.replace("self_attn.sdpa.rel_k_embed",
                            "self_attn.rel_k_embed")
        name = name.replace("conv.pointwise_conv1",
                            "conv_module.pointwise_conv1")
        name = name.replace("conv.depthwise_conv",
                            "conv_module.depthwise_conv")
        name = name.replace("conv.pointwise_conv2",
                            "conv_module.pointwise_conv2")
        name = name.replace("conv.layer_norm", "conv_module.norm")
        name = name.replace("ffn1.inner_proj", "feed_forward_macaron.w_1")
        name = name.replace("ffn1.output_proj", "feed_forward_macaron.w_2")
        name = name.replace("ffn2.inner_proj", "feed_forward.w_1")
        name = name.replace("ffn2.output_proj", "feed_forward.w_2")
        name = name.replace("encoder_frontend.model_dim_proj",
                            "encoder.embed.out")
        name = name.replace("encoder_frontend.post_extract_layer_norm",
                            "encoder.embed.norm")
        name = name.replace(".layer_norm.", ".norm_final.")
        wenet_state_dict[name] = conformer_state_dict[old_name]

    print("Saving fp32 ckpt to {}...".format(wenet_state_dict_path))
    torch.save(wenet_state_dict, wenet_state_dict_path)
    print(
        "DONE\n===================- End CKPT Conversion ====================\n"
    )


def get_args():
    parser = argparse.ArgumentParser(
        description='load and parse w2vbert2-conformer')
    # yapf: disable
    parser.add_argument(
        '--w2vbert2_ckpt',
        required=True,
        help= 'https://huggingface.co/facebook/conformer-shaw/resolve/main/conformer_shaw.pt' # noqa
    )
    # yapf: enable
    parser.add_argument('--output_dir',
                        default='.',
                        help='output file in wenet\'s style: ' +
                        'units.txt, train.yaml, model.pt')
    args = parser.parse_args()
    return args


def main():

    args = get_args()
    args.jit = True
    checkpoint = torch.load(args.w2vbert2_ckpt, map_location="cpu")

    os.makedirs(args.output_dir, exist_ok=True)
    convert_to_wenet_state_dict(
        checkpoint["model"],
        os.path.join(args.output_dir, 'wenet_w2vbert_conformer_600m.pt'))


if __name__ == '__main__':
    main()
