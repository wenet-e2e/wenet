import pytest
import torch

from wenet.models.transformer.search import ctc_prefix_beam_search
from wenet.utils.context_graph import ContextGraph

SYMBOL_TABLE = {'北': 1, '京': 2, '大': 3, '学': 4, '天': 5, '背': 6, '景': 7}


def build_graph(tmp_path, phrases):
    context_list = tmp_path / 'context.txt'
    context_list.write_text('\n'.join(phrases) + '\n', encoding='utf8')
    return ContextGraph(str(context_list), SYMBOL_TABLE, context_score=3.0)


def decode(graph, text):
    state = graph.root
    total = 0.0
    for ch in text:
        score, state = graph.forward_one_step(state, SYMBOL_TABLE[ch])
        total += score
    score, _ = graph.finalize(state)
    return total + score


@pytest.mark.parametrize('phrases', [
    ['北京', '北京大学'],
    ['北京大学', '北京'],
])
def test_phrase_score_does_not_depend_on_list_order(tmp_path, phrases):
    graph = build_graph(tmp_path, phrases)

    assert decode(graph, '北京天') == pytest.approx(6.0)
    assert decode(graph, '北京') == pytest.approx(6.0)
    assert decode(graph, '北京大学') == pytest.approx(18.0)


@pytest.mark.parametrize('phrases', [
    ['北京', '北京大学'],
    ['北京大学', '北京'],
])
def test_ctc_prefix_beam_search_is_biased_in_either_order(tmp_path, phrases):
    graph = build_graph(tmp_path, phrases)
    # 背景 is a little likelier than 北京 on the acoustic side
    probs = torch.full((1, 3, len(SYMBOL_TABLE) + 1), 1e-6)
    probs[0, 0, SYMBOL_TABLE['背']] = 0.55
    probs[0, 0, SYMBOL_TABLE['北']] = 0.45
    probs[0, 1, SYMBOL_TABLE['景']] = 0.55
    probs[0, 1, SYMBOL_TABLE['京']] = 0.45
    probs[0, 2, SYMBOL_TABLE['天']] = 1.0

    hyp = ctc_prefix_beam_search(probs.log(),
                                 torch.tensor([3]),
                                 beam_size=5,
                                 context_graph=graph)[0]

    expected = [SYMBOL_TABLE[ch] for ch in '北京天']
    assert list(hyp.tokens) == expected


@pytest.mark.parametrize('phrases', [
    ['北京', '北京大', '天北京'],
    ['北京大', '北京', '天北京'],
    ['北京大', '天北京', '北京'],
])
def test_phrase_inside_another_phrase_is_scored_through_the_fail_arc(
        tmp_path, phrases):
    graph = build_graph(tmp_path, phrases)

    assert decode(graph, '天北京') == pytest.approx(15.0)


def test_phrase_listed_twice_is_scored_once(tmp_path):
    graph = build_graph(tmp_path, ['北京', '北京'])

    assert decode(graph, '北京') == pytest.approx(6.0)


def test_blank_line_does_not_mark_the_root_as_a_phrase_end(tmp_path):
    graph = build_graph(tmp_path, ['北京', '', '北京大学'])

    assert not graph.root.is_end
    assert decode(graph, '北京') == pytest.approx(6.0)
