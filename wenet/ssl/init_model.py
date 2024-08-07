from wenet.ssl.bestrq.bestrq_model import BestRQModel
from wenet.ssl.wav2vec2.wav2vec2_model import Wav2vec2Model
from wenet.ssl.w2vbert.w2vbert_model import W2VBERTModel


def init_model(configs, encoder):

    assert 'model' in configs
    model_type = configs['model']
    ssl_model_class = {
        "w2vbert_model": W2VBERTModel,
        "wav2vec_model": Wav2vec2Model,
        "bestrq_model": BestRQModel
    }
    assert model_type in ssl_model_class.keys()
    model = ssl_model_class[model_type](encoder=encoder,
                                        **configs['model_conf'])
    return model, configs
