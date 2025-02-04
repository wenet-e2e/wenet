from wenet.ssl.bestrq.bestrq_model import BestRQModel
from wenet.ssl.wav2vec2.wav2vec2_model import Wav2vec2Model
from wenet.ssl.w2vbert.w2vbert_model import W2VBERTModel

WENET_SSL_MODEL_CLASS = {
    "w2vbert_model": W2VBERTModel,
    "wav2vec_model": Wav2vec2Model,
    "bestrq_model": BestRQModel
}


def init_model(configs, encoder):

    assert 'model' in configs
    model_type = configs['model']
    assert model_type in WENET_SSL_MODEL_CLASS.keys()
    model = WENET_SSL_MODEL_CLASS[model_type](encoder=encoder,
                                              **configs['model_conf'])
    return model
