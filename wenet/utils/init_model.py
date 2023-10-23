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
from wenet.paraformer.ali_paraformer.model import SanmDecoer, SanmEncoder
from wenet.transducer.joint import TransducerJoint
from wenet.transducer.predictor import (ConvPredictor, EmbeddingPredictor,
                                        RNNPredictor)
from wenet.transducer.transducer import Transducer
from wenet.transformer.asr_model import ASRModel
from wenet.transformer.cmvn import GlobalCMVN
from wenet.transformer.ctc import CTC
from wenet.transformer.decoder import BiTransformerDecoder, TransformerDecoder
from wenet.transformer.encoder import ConformerEncoder, TransformerEncoder
from wenet.branchformer.encoder import BranchformerEncoder
from wenet.e_branchformer.encoder import EBranchformerEncoder
from wenet.squeezeformer.encoder import SqueezeformerEncoder
from wenet.efficient_conformer.encoder import EfficientConformerEncoder
from wenet.paraformer.paraformer import Paraformer
from wenet.paraformer.ali_paraformer.model import AliParaformer
from wenet.cif.predictor import Predictor
from wenet.utils.cmvn import load_cmvn


def init_model(configs):
    if configs['cmvn_file'] is not None:
        mean, istd = load_cmvn(configs['cmvn_file'], configs['is_json_cmvn'])
        global_cmvn = GlobalCMVN(
            torch.from_numpy(mean).float(),
            torch.from_numpy(istd).float())
    else:
        global_cmvn = None

    input_dim = configs['input_dim']
    vocab_size = configs['output_dim']

    encoder_type = configs.get('encoder', 'conformer')
    decoder_type = configs.get('decoder', 'bitransformer')

    if encoder_type == 'conformer':
        encoder = ConformerEncoder(input_dim,
                                   global_cmvn=global_cmvn,
                                   **configs['encoder_conf'])
    elif encoder_type == 'squeezeformer':
        encoder = SqueezeformerEncoder(input_dim,
                                       global_cmvn=global_cmvn,
                                       **configs['encoder_conf'])
    elif encoder_type == 'efficientConformer':
        encoder = EfficientConformerEncoder(
            input_dim,
            global_cmvn=global_cmvn,
            **configs['encoder_conf'],
            **configs['encoder_conf']['efficient_conf']
            if 'efficient_conf' in configs['encoder_conf'] else {})
    elif encoder_type == 'branchformer':
        encoder = BranchformerEncoder(input_dim,
                                      global_cmvn=global_cmvn,
                                      **configs['encoder_conf'])
    elif encoder_type == 'e_branchformer':
        encoder = EBranchformerEncoder(input_dim,
                                       global_cmvn=global_cmvn,
                                       **configs['encoder_conf'])
    elif encoder_type == 'SanmEncoder':
        assert 'lfr_conf' in configs
        encoder = SanmEncoder(global_cmvn=global_cmvn,
                              input_size=configs['lfr_conf']['lfr_m'] *
                              input_dim,
                              **configs['encoder_conf'])
    else:
        encoder = TransformerEncoder(input_dim,
                                     global_cmvn=global_cmvn,
                                     **configs['encoder_conf'])
    if decoder_type == 'transformer':
        decoder = TransformerDecoder(vocab_size, encoder.output_size(),
                                     **configs['decoder_conf'])
    elif decoder_type == 'SanmDecoder':
        assert isinstance(encoder, SanmEncoder)
        decoder = SanmDecoer(vocab_size=vocab_size,
                             encoder_output_size=encoder.output_size(),
                             **configs['decoder_conf'])
    else:
        assert 0.0 < configs['model_conf']['reverse_weight'] < 1.0
        assert configs['decoder_conf']['r_num_blocks'] > 0
        decoder = BiTransformerDecoder(vocab_size, encoder.output_size(),
                                       **configs['decoder_conf'])
    ctc = CTC(vocab_size, encoder.output_size())

    # Init joint CTC/Attention or Transducer model
    if 'predictor' in configs:
        predictor_type = configs.get('predictor', 'rnn')
        if predictor_type == 'rnn':
            predictor = RNNPredictor(vocab_size, **configs['predictor_conf'])
        elif predictor_type == 'embedding':
            predictor = EmbeddingPredictor(vocab_size,
                                           **configs['predictor_conf'])
            configs['predictor_conf']['output_size'] = configs[
                'predictor_conf']['embed_size']
        elif predictor_type == 'conv':
            predictor = ConvPredictor(vocab_size, **configs['predictor_conf'])
            configs['predictor_conf']['output_size'] = configs[
                'predictor_conf']['embed_size']
        else:
            raise NotImplementedError(
                "only rnn, embedding and conv type support now")
        configs['joint_conf']['enc_output_size'] = configs['encoder_conf'][
            'output_size']
        configs['joint_conf']['pred_output_size'] = configs['predictor_conf'][
            'output_size']
        joint = TransducerJoint(vocab_size, **configs['joint_conf'])
        model = Transducer(vocab_size=vocab_size,
                           blank=0,
                           predictor=predictor,
                           encoder=encoder,
                           attention_decoder=decoder,
                           joint=joint,
                           ctc=ctc,
                           **configs['model_conf'])
    elif 'paraformer' in configs:
        predictor = Predictor(**configs['cif_predictor_conf'])
        if isinstance(encoder, SanmEncoder):
            assert isinstance(decoder, SanmDecoer)
            # NOTE(Mddct): only support inference for now
            print('hello world')
            model = AliParaformer(encoder, decoder, predictor)
        else:
            model = Paraformer(vocab_size=vocab_size,
                               encoder=encoder,
                               decoder=decoder,
                               ctc=ctc,
                               predictor=predictor,
                               **configs['model_conf'])
    else:
        model = ASRModel(vocab_size=vocab_size,
                         encoder=encoder,
                         decoder=decoder,
                         ctc=ctc,
                         lfmmi_dir=configs.get('lfmmi_dir', ''),
                         **configs['model_conf'])
    return model
