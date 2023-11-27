from wenet.cli.model import load_model  # noqa
from wenet.transformer.subsampling import (
    LinearNoSubsampling, EmbedinigNoSubsampling,
    Conv1dSubsampling2, Conv2dSubsampling4,
    Conv2dSubsampling6, Conv2dSubsampling8,
)
from wenet.efficient_conformer.subsampling import Conv2dSubsampling2
from wenet.squeezeformer.subsampling import DepthwiseConv2dSubsampling4
from wenet.transformer.embedding import (
    PositionalEncoding, RelPositionalEncoding,
    WhisperPositionalEncoding, LearnablePositionalEncoding,
    NoPositionalEncoding
)
from wenet.transformer.attention import (
    MultiHeadedAttention, RelPositionMultiHeadedAttention
)


WENET_SUBSAMPLE_CLASSES = {
    "linear": LinearNoSubsampling,
    "embed": EmbedinigNoSubsampling,
    "conv1d2": Conv1dSubsampling2,
    "conv2d2": Conv2dSubsampling2,
    "conv2d": Conv2dSubsampling4,
    "dwconv2d4": DepthwiseConv2dSubsampling4,
    "conv2d6": Conv2dSubsampling6,
    "conv2d8": Conv2dSubsampling8,
}

WENET_EMB_CLASSES = {
    "abs_pos": PositionalEncoding,
    "rel_pos": RelPositionalEncoding,
    "no_pos": NoPositionalEncoding,
    "abs_pos_whisper": WhisperPositionalEncoding,
    "embed_learnable_pe": LearnablePositionalEncoding,
    "none": NoPositionalEncoding,
}

WENET_ATTENTION_CLASSES = {
    "selfattn": MultiHeadedAttention,
    "rel_selfattn": RelPositionMultiHeadedAttention,
}
