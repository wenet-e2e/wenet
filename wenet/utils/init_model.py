# Copyright (c) 2022 Binbin Zhang (binbzha@qq.com)
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

import torch

from wenet.k2.model import K2Model
from wenet.paraformer.cif import Cif
from wenet.paraformer.layers import SanmDecoder, SanmEncoder
from wenet.paraformer.paraformer import Paraformer, Predictor
from wenet.transducer.joint import TransducerJoint
from wenet.transducer.predictor import (ConvPredictor, EmbeddingPredictor,
                                        RNNPredictor)
from wenet.transducer.transducer import Transducer
from wenet.transformer.asr_model import ASRModel
from wenet.transformer.cmvn import GlobalCMVN
from wenet.transformer.ctc import CTC
from wenet.transformer.encoder import TransformerEncoder, ConformerEncoder
from wenet.transformer.decoder import BiTransformerDecoder, TransformerDecoder
from wenet.branchformer.encoder import BranchformerEncoder
from wenet.e_branchformer.encoder import EBranchformerEncoder
from wenet.squeezeformer.encoder import SqueezeformerEncoder
from wenet.efficient_conformer.encoder import EfficientConformerEncoder
from wenet.ctl_model.encoder import DualTransformerEncoder, DualConformerEncoder
from wenet.ctl_model.asr_model_ctl import CTLModel
from wenet.whisper.whisper import Whisper
from wenet.utils.cmvn import load_cmvn
from wenet.utils.checkpoint import load_checkpoint, load_trained_modules

WENET_ENCODER_CLASSES = {
    "transformer": TransformerEncoder,
    "conformer": ConformerEncoder,
    "squeezeformer": SqueezeformerEncoder,
    "efficientConformer": EfficientConformerEncoder,
    "branchformer": BranchformerEncoder,
    "e_branchformer": EBranchformerEncoder,
    "dual_transformer": DualTransformerEncoder,
    "dual_conformer": DualConformerEncoder,
    'sanm_encoder': SanmEncoder,
}

WENET_DECODER_CLASSES = {
    "transformer": TransformerDecoder,
    "bitransformer": BiTransformerDecoder,
    "sanm_decoder": SanmDecoder,
}

WENET_CTC_CLASSES = {
    "ctc": CTC,
}

WENET_PREDICTOR_CLASSES = {
    "rnn": RNNPredictor,
    "embedding": EmbeddingPredictor,
    "conv": ConvPredictor,
    "cif_predictor": Cif,
    "paraformer_predictor": Predictor,
}

WENET_JOINT_CLASSES = {
    "transducer_joint": TransducerJoint,
}

WENET_MODEL_CLASSES = {
    "asr_model": ASRModel,
    "ctl_model": CTLModel,
    "whisper": Whisper,
    "k2_model": K2Model,
    "transducer": Transducer,
    'paraformer': Paraformer,
}


def init_model(args, configs):

    # TODO(xcsong): Forcefully read the 'cmvn' attribute.
    if configs.get('cmvn', None) == 'global_cmvn':
        mean, istd = load_cmvn(configs['cmvn_conf']['cmvn_file'],
                               configs['cmvn_conf']['is_json_cmvn'])
        global_cmvn = GlobalCMVN(
            torch.from_numpy(mean).float(),
            torch.from_numpy(istd).float())
    else:
        global_cmvn = None

    input_dim = configs['input_dim']
    vocab_size = configs['output_dim']

    encoder_type = configs.get('encoder', 'conformer')
    decoder_type = configs.get('decoder', 'bitransformer')
    ctc_type = configs.get('ctc', 'ctc')

    encoder = WENET_ENCODER_CLASSES[encoder_type](
        input_dim,
        global_cmvn=global_cmvn,
        **configs['encoder_conf'],
        **configs['encoder_conf']['efficient_conf']
        if 'efficient_conf' in configs['encoder_conf'] else {})

    decoder = WENET_DECODER_CLASSES[decoder_type](vocab_size,
                                                  encoder.output_size(),
                                                  **configs['decoder_conf'])

    ctc = WENET_CTC_CLASSES[ctc_type](
        vocab_size,
        encoder.output_size(),
        blank_id=configs['ctc_conf']['ctc_blank_id']
        if 'ctc_conf' in configs else 0)

    model_type = configs.get('model', 'asr_model')
    if model_type == "transducer":
        predictor_type = configs.get('predictor', 'rnn')
        joint_type = configs.get('joint', 'transducer_joint')
        predictor = WENET_PREDICTOR_CLASSES[predictor_type](
            vocab_size, **configs['predictor_conf'])
        joint = WENET_JOINT_CLASSES[joint_type](vocab_size,
                                                **configs['joint_conf'])
        model = WENET_MODEL_CLASSES[model_type](
            vocab_size=vocab_size,
            blank=0,
            predictor=predictor,
            encoder=encoder,
            attention_decoder=decoder,
            joint=joint,
            ctc=ctc,
            special_tokens=configs.get('tokenizer_conf',
                                       {}).get('special_tokens', None),
            **configs['model_conf'])
    elif model_type == 'paraformer':
        predictor_type = configs.get('predictor', 'cif')
        predictor = WENET_PREDICTOR_CLASSES[predictor_type](
            **configs['predictor_conf'])
        model = WENET_MODEL_CLASSES[model_type](
            vocab_size=vocab_size,
            encoder=encoder,
            decoder=decoder,
            predictor=predictor,
            ctc=ctc,
            **configs['model_conf'],
            special_tokens=configs.get('tokenizer_conf',
                                       {}).get('special_tokens', None),
        )
    else:
        model = WENET_MODEL_CLASSES[model_type](
            vocab_size=vocab_size,
            encoder=encoder,
            decoder=decoder,
            ctc=ctc,
            special_tokens=configs.get('tokenizer_conf',
                                       {}).get('special_tokens', None),
            **configs['model_conf'])

    # If specify checkpoint, load some info from checkpoint
    if hasattr(args, 'checkpoint') and args.checkpoint is not None:
        infos = load_checkpoint(model, args.checkpoint)
    elif hasattr(args, 'enc_init') and args.enc_init is not None:
        infos = load_trained_modules(model, args)
    else:
        infos = {}
    configs["init_infos"] = infos
    print(configs)

    # Tie emb.weight to decoder.output_layer.weight
    if model.decoder.tie_word_embedding:
        model.decoder.tie_or_clone_weights(jit_mode=args.jit)

    return model, configs
