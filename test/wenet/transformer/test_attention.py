import torch
import pytest
from wenet.transformer.attention import (MultiHeadedAttention,
                                         RelPositionMultiHeadedAttention,
                                         ShawRelPositionMultiHeadedAttention)
from wenet.transformer.embedding import RelPositionalEncoding
from wenet.transformer.encoder_layer import (ConformerEncoderLayer,
                                             TransformerEncoderLayer)
from wenet.transformer.positionwise_feed_forward import PositionwiseFeedForward
from wenet.utils.class_utils import WENET_ACTIVATION_CLASSES

from wenet.utils.mask import add_optional_chunk_mask, make_non_pad_mask


@pytest.mark.parametrize("args", [
    {
        "n_feat": 256,
        "n_head": 4,
        "dropout_rate": 0.0
    },
    {
        "n_feat": 512,
        "n_head": 8,
        "dropout_rate": 0.0
    },
    {
        "n_feat": 1280,
        "n_head": 20,
        "dropout_rate": 0.0
    },
    {
        "n_feat": 512,
        "n_head": 4,
        "dropout_rate": 0.0
    },
])
def test_multi_head_attention_sdpa(args):
    torch.manual_seed(777)
    mha_module = MultiHeadedAttention(use_sdpa=False, **args)
    torch.manual_seed(777)
    mha_module_with_sdpa = MultiHeadedAttention(use_sdpa=True, **args)
    mha_module.eval()
    mha_module_with_sdpa.eval()

    q = torch.rand(10, 100, args['n_feat'])
    k = torch.rand(10, 100, args['n_feat'])
    v = torch.rand(10, 100, args['n_feat'])
    input_lens = torch.tensor([100, 90, 80, 79, 60, 51, 40, 30, 10, 5])
    mask = make_non_pad_mask(input_lens).unsqueeze(1)
    att_mask = add_optional_chunk_mask(q,
                                       mask,
                                       use_dynamic_chunk=True,
                                       decoding_chunk_size=0,
                                       static_chunk_size=0,
                                       use_dynamic_left_chunk=True,
                                       num_decoding_left_chunks=-1)
    output, cache = mha_module(q, k, v, mask=att_mask)

    att_mask_bias = (1.0 - att_mask.float()) * torch.finfo(torch.float).min
    output_with_sdpa, cache_with_sdpa = mha_module_with_sdpa(
        q, k, v, mask=att_mask_bias)
    assert torch.allclose(
        output * mask.transpose(1, 2),
        output_with_sdpa * mask.transpose(1, 2),
        atol=9e-7,
    )
    assert torch.allclose(cache[0], cache_with_sdpa[0])
    assert torch.allclose(cache[1], cache_with_sdpa[1])

    n_blocks = 12
    torch.manual_seed(777)
    mha_layers = [
        TransformerEncoderLayer(
            args['n_feat'],
            MultiHeadedAttention(use_sdpa=False, **args),
            PositionwiseFeedForward(
                args['n_feat'],
                2048,
                0.0,
                WENET_ACTIVATION_CLASSES['swish'](),
            ),
            0.0,
            normalize_before=True,
        ) for _ in range(n_blocks)
    ]

    torch.manual_seed(777)
    mha_layers_with_sdpa = [
        TransformerEncoderLayer(
            args['n_feat'],
            MultiHeadedAttention(use_sdpa=True, **args),
            PositionwiseFeedForward(
                args['n_feat'],
                2048,
                0.0,
                WENET_ACTIVATION_CLASSES['swish'](),
            ),
            0.0,
            normalize_before=True,
        ) for _ in range(n_blocks)
    ]

    for i in range(n_blocks):
        output, _, cache, _ = mha_layers[i](q, att_mask, None, mask)
        output_with_sdpa, _, cache_with_sdpa, _ = mha_layers_with_sdpa[i](
            q, att_mask_bias, None, mask)

        assert torch.allclose(
            output * mask.transpose(1, 2),
            output_with_sdpa * mask.transpose(1, 2),
            atol=9e-7,
            rtol=9e-4,
        )
        assert torch.allclose(cache[0], cache_with_sdpa[0])
        assert torch.allclose(cache[1], cache_with_sdpa[1])

        q = output


@pytest.mark.parametrize("args", [
    {
        "n_feat": 256,
        "n_head": 4,
        "dropout_rate": 0.0
    },
    {
        "n_feat": 512,
        "n_head": 8,
        "dropout_rate": 0.0
    },
    {
        "n_feat": 1280,
        "n_head": 20,
        "dropout_rate": 0.0
    },
    {
        "n_feat": 512,
        "n_head": 4,
        "dropout_rate": 0.0
    },
])
def test_rel_position_multi_head_attention_sdpa(args):
    rel_pos_moduls = RelPositionalEncoding(args['n_feat'], dropout_rate=0.0)
    torch.manual_seed(777)
    rel_mha_module = RelPositionMultiHeadedAttention(use_sdpa=False, **args)
    torch.manual_seed(777)
    rel_mha_module_with_sdpa = RelPositionMultiHeadedAttention(use_sdpa=True,
                                                               **args)
    rel_mha_module.eval()
    rel_mha_module_with_sdpa.eval()

    q = torch.rand(10, 100, args['n_feat'])
    _, pos_emb = rel_pos_moduls(q)
    k = torch.rand(10, 100, args['n_feat'])
    v = torch.rand(10, 100, args['n_feat'])
    input_lens = torch.tensor([100, 90, 80, 79, 60, 51, 40, 30, 10, 5])
    mask = make_non_pad_mask(input_lens).unsqueeze(1)
    att_mask = add_optional_chunk_mask(q,
                                       mask,
                                       use_dynamic_chunk=True,
                                       decoding_chunk_size=0,
                                       static_chunk_size=0,
                                       use_dynamic_left_chunk=True,
                                       num_decoding_left_chunks=-1)
    output, cache = rel_mha_module(q, k, v, mask=att_mask, pos_emb=pos_emb)

    att_mask_bias = (1.0 - att_mask.float()) * torch.finfo(torch.float).min
    output_with_sdpa, cache_with_sdpa = rel_mha_module_with_sdpa(
        q, k, v, mask=att_mask_bias, pos_emb=pos_emb)
    assert torch.allclose(
        output * mask.transpose(1, 2),
        output_with_sdpa * mask.transpose(1, 2),
        atol=9e-7,
    )
    assert torch.allclose(cache[0], cache_with_sdpa[0])
    assert torch.allclose(cache[1], cache_with_sdpa[1])

    n_blocks = 12
    torch.manual_seed(777)
    rel_mha_layers = [
        ConformerEncoderLayer(
            args['n_feat'],
            RelPositionMultiHeadedAttention(use_sdpa=False, **args),
            PositionwiseFeedForward(
                args['n_feat'],
                2048,
                0.0,
                WENET_ACTIVATION_CLASSES['swish'](),
            ),
            None,
            None,
            0.0,
            normalize_before=True,
        ) for _ in range(n_blocks)
    ]

    torch.manual_seed(777)
    rel_mha_layers_with_sdpa = [
        ConformerEncoderLayer(
            args['n_feat'],
            RelPositionMultiHeadedAttention(use_sdpa=True, **args),
            PositionwiseFeedForward(
                args['n_feat'],
                2048,
                0.0,
                WENET_ACTIVATION_CLASSES['swish'](),
            ),
            None,
            None,
            0.0,
            normalize_before=True,
        ) for _ in range(n_blocks)
    ]

    for i in range(n_blocks):
        output, _, cache, _ = rel_mha_layers[i](q, att_mask, pos_emb, mask)
        output_with_sdpa, _, cache_with_sdpa, _ = rel_mha_layers_with_sdpa[i](
            q, att_mask_bias, pos_emb, mask)

        assert torch.allclose(
            output * mask.transpose(1, 2),
            output_with_sdpa * mask.transpose(1, 2),
            atol=9e-7,
            rtol=9e-4,
        )
        assert torch.allclose(cache[0], cache_with_sdpa[0])
        assert torch.allclose(cache[1], cache_with_sdpa[1])
        q = output


def test_shaw_rel_position_multihead_attention():
    torch.manual_seed(777)
    module = ShawRelPositionMultiHeadedAttention(8, 256, 0.0, use_sdpa=False)

    torch.manual_seed(777)
    module_sdpa = ShawRelPositionMultiHeadedAttention(8,
                                                      256,
                                                      0.0,
                                                      use_sdpa=True)
    q = torch.rand(2, 10, 256)
    k = torch.rand(2, 10, 256)
    v = torch.rand(2, 10, 256)
    pos_emb = torch.zeros(0, 0, 0)
    mask = torch.ones(2, 10, 10)
    out, _ = module(q, k, v, mask, pos_emb)
    out_sdpa, _ = module_sdpa(q, k, v, mask, pos_emb)

    torch.allclose(out, out_sdpa)
