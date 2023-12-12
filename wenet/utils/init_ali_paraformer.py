import torch
from wenet.paraformer.cif import Cif
from wenet.paraformer.layers import (SanmDecoder, SanmEncoder)
from wenet.paraformer.paraformer import Paraformer
from wenet.transformer.cmvn import GlobalCMVN
from wenet.utils.checkpoint import load_checkpoint
from wenet.utils.cmvn import load_cmvn


def init_model(configs, checkpoint_path=None):
    mean, istd = load_cmvn(configs['cmvn_file'], True)
    global_cmvn = GlobalCMVN(
        torch.from_numpy(mean).float(),
        torch.from_numpy(istd).float())
    input_dim = configs['input_dim']
    vocab_size = configs['output_dim']
    encoder = SanmEncoder(global_cmvn=global_cmvn,
                          input_size=configs['lfr_conf']['lfr_m'] * input_dim,
                          **configs['encoder_conf'])
    decoder = SanmDecoder(vocab_size=vocab_size,
                          encoder_output_size=encoder.output_size(),
                          **configs['decoder_conf'])
    predictor = Cif(**configs['cif_predictor_conf'])
    model = Paraformer(
        vocab_size=vocab_size,
        encoder=encoder,
        decoder=decoder,
        predictor=predictor,
        **configs['model_conf'],
    )

    if checkpoint_path is not None:
        load_checkpoint(model, checkpoint_path)
    return model, configs
