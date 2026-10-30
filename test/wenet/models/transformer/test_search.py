import pytest
import torch

from wenet.models.transformer.search import (attention_rescoring,
                                             ctc_prefix_beam_search)
from wenet.utils.context_graph import ContextGraph

SYMBOL_TABLE = {'北': 1, '京': 2, '大': 3, '学': 4, '天': 5, '背': 6, '景': 7}
VOCAB_SIZE = len(SYMBOL_TABLE) + 3  # blank, sos, eos
CONTEXT_SCORE = 3.0
# 背景 is a little likelier than 北京 on the acoustic side
AMBIGUOUS_FRAMES = [{'背': 0.55, '北': 0.45}, {'景': 0.55, '京': 0.45}]


class UniformDecoderModel:
    """ The decoder has no opinion, only the ctc score decides """
    ignore_id = -1

    def sos_symbol(self):
        return VOCAB_SIZE - 2

    def eos_symbol(self):
        return VOCAB_SIZE - 1

    def forward_attention_decoder(self, hyps, hyps_lens, encoder_out,
                                  reverse_weight):
        return torch.zeros(hyps.size(0), hyps.size(1),
                           VOCAB_SIZE), torch.empty(0)


def build_graph(tmp_path, phrases):
    context_list = tmp_path / 'context.txt'
    context_list.write_text('\n'.join(phrases) + '\n', encoding='utf8')
    return ContextGraph(str(context_list),
                        SYMBOL_TABLE,
                        context_score=CONTEXT_SCORE)


def beam_search(frames, graph=None):
    probs = torch.full((1, len(frames), VOCAB_SIZE), 1e-6)
    for t, frame in enumerate(frames):
        for ch, prob in frame.items():
            probs[0, t, SYMBOL_TABLE[ch]] = prob
    return ctc_prefix_beam_search(probs.log(),
                                  torch.tensor([len(frames)]),
                                  beam_size=5,
                                  context_graph=graph)


def tokens(text):
    return [SYMBOL_TABLE[ch] for ch in text]


@pytest.mark.parametrize('frames', [
    [{'北': 1.0}, {'京': 1.0}],
    [{'北': 1.0}, {'京': 1.0}, {'天': 1.0}],
])
def test_matched_phrase_keeps_its_bonus(tmp_path, frames):
    graph = build_graph(tmp_path, ['北京'])

    plain = beam_search(frames)[0]
    biased = beam_search(frames, graph)[0]

    assert biased.tokens == plain.tokens
    assert biased.score == pytest.approx(plain.score + 2 * CONTEXT_SCORE)


def test_unfinished_phrase_does_not_beat_a_likelier_hypothesis(tmp_path):
    graph = build_graph(tmp_path, ['北京大学'])

    plain = beam_search(AMBIGUOUS_FRAMES)[0]
    biased = beam_search(AMBIGUOUS_FRAMES, graph)[0]

    assert list(biased.tokens) == tokens('背景')
    assert biased.score == pytest.approx(plain.score)


def test_attention_rescoring_sees_the_bonus(tmp_path):
    graph = build_graph(tmp_path, ['北京'])
    results = beam_search(AMBIGUOUS_FRAMES, graph)

    best = attention_rescoring(UniformDecoderModel(),
                               results,
                               torch.zeros(1, 2, 4),
                               torch.tensor([2]),
                               ctc_weight=0.5)[0]

    assert list(best.tokens) == tokens('北京')
