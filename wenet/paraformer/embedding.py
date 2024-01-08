from wenet.transformer.embedding import WhisperPositionalEncoding


class ParaformerPositinoalEncoding(WhisperPositionalEncoding):
    """ Sinusoids position encoding used in paraformer.encoder
    """

    def __init__(self,
                 depth: int,
                 d_model: int,
                 dropout_rate: float = 0.1,
                 max_len: int = 1500):
        super().__init__(depth, dropout_rate, max_len)
        self.xscale = d_model**0.5
