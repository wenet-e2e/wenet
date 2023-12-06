"""NER
Input (e.g. with_hotwords_ali.log):
BAC009S0764W0179    国务院发展研究中心市场经济研究所副所长邓郁松认为
BAC009S0764W0205    本报记者王颖春国家发改委近日发出通知

Outpus (e.g. with_hotwords_ali.log.ner):
BAC009S0764W0179    国务院发展研究中心市场经济研究所副所长邓郁松认为    邓郁松
BAC009S0764W0205    本报记者王颖春国家发改委近日发出通知  王颖春

Run:
python compute_hotwords_f1.py \
    --label="data/aishell1_text_hotwords" \
    --preds="data/with_hotwords_ali.log;data/without_hotwords_ali.log" \
    --hotword="data/hotwords.yaml"
"""


def _sorted_iteritems(d):
    return sorted(d.items())


def _iteritems(d):
    return iter(d.items())


def _iterkeys(d):
    return iter(d.keys())


_basestring = str
_SENTINEL = object()

import collections
try:
    collectionsAbc = collections.abc
except AttributeError:
    collectionsAbc = collections

import argparse
import os
import yaml


class _Node(object):
    """A single node of a trie.
    """
    __slots__ = ('children', 'value')

    def __init__(self):
        self.children = {}
        self.value = _SENTINEL

    def iterate(self, path, shallow, iteritems):
        """Yields all the nodes with values associated to them in the trie.
        """
        # Use iterative function with stack on the heap so we don't hit Python's
        # recursion depth limits.
        node = self
        stack = []
        while True:
            if node.value is not _SENTINEL:
                yield path, node.value

            if (not shallow or node.value is _SENTINEL) and node.children:
                stack.append(iter(iteritems(node.children)))
                path.append(None)

            while True:
                try:
                    step, node = next(stack[-1])
                    path[-1] = step
                    break
                except StopIteration:
                    stack.pop()
                    path.pop()
                except IndexError:
                    return

    def __eq__(self, other):
        # Like iterate, we don't recurse so this works on deep tries.
        a, b = self, other
        stack = []
        while True:
            if a.value != b.value or len(a.children) != len(b.children):
                return False
            if a.children:
                stack.append((_iteritems(a.children), b.children))

            while True:
                try:
                    key, a = next(stack[-1][0])
                    b = stack[-1][1].get(key)
                    if b is None:
                        return False
                    break
                except StopIteration:
                    stack.pop()
                except IndexError:
                    return True
        return self.value == other.value and self.children == other.children

    def __ne__(self, other):
        return not self.__eq__(other)

    def __bool__(self):
        return bool(self.value is not _SENTINEL or self.children)

    __nonzero__ = __bool__

    __hash__ = None

    def __getstate__(self):
        """Get state used for pickling.
        """
        # Like iterate, we don't recurse so pickling works on deep tries.
        state = [] if self.value is _SENTINEL else [0]
        last_cmd = 0
        node = self
        stack = []
        while True:
            if node.value is not _SENTINEL:
                last_cmd = 0
                state.append(node.value)
            stack.append(_iteritems(node.children))

            while True:
                try:
                    step, node = next(stack[-1])
                except StopIteration:
                    if last_cmd < 0:
                        state[-1] -= 1
                    else:
                        last_cmd = -1
                        state.append(-1)
                    stack.pop()
                    continue
                except IndexError:
                    if last_cmd < 0:
                        state.pop()
                    return state

                if last_cmd > 0:
                    last_cmd += 1
                    state[-last_cmd] += 1
                else:
                    last_cmd = 1
                    state.append(1)
                state.append(step)
                break

    def __setstate__(self, state):
        """Unpickles node.  See :func:`_Node.__getstate__`."""
        self.__init__()
        state = iter(state)
        stack = [self]
        for cmd in state:
            if cmd < 0:
                del stack[cmd:]
            else:
                while cmd > 0:
                    stack.append(type(self)())
                    stack[-2].children[next(state)] = stack[-1]
                    cmd -= 1
                stack[-1].value = next(state)


class CharTrie(collectionsAbc.MutableMapping):
    """A trie implementation with dict interface plus some extensions.
    """

    def __init__(self, *args, **kwargs):
        self._root = _Node()
        self._sorted = False
        self.update(*args, **kwargs)

    @property
    def _iteritems(self):
        return _sorted_iteritems if self._sorted else _iteritems

    def enable_sorting(self, enable=True):
        """Enables sorting of child nodes when iterating and traversing.
        """
        self._sorted = enable

    def clear(self):
        """Removes all the values from the trie."""
        self._root = _Node()

    def update(self, *args, **kwargs):
        """Updates stored values.  Works like :func:`dict.update`."""
        if len(args) > 1:
            raise ValueError('update() takes at most one positional argument, '
                             '%d given.' % len(args))
        if args and isinstance(args[0], CharTrie):
            for key, value in _iteritems(args[0]):
                self[key] = value
            args = ()
        super(CharTrie, self).update(*args, **kwargs)

    def copy(self):
        """Returns a shallow copy of the trie."""
        return self.__class__(self)

    @classmethod
    def fromkeys(cls, keys, value=None):
        """Creates a new trie with given keys set.
        """
        trie = cls()
        for key in keys:
            trie[key] = value
        return trie

    def _get_node(self, key, create=False):
        """Returns node for given key.  Creates it if requested.
        """
        node = self._root
        trace = [(None, node)]
        for step in self.__path_from_key(key):
            if create:
                node = node.children.setdefault(step, _Node())
            else:
                node = node.children.get(step)
                if not node:
                    raise KeyError(key)
            trace.append((step, node))
        return node, trace

    def __iter__(self):
        return self.iterkeys()

    def iteritems(self, prefix=_SENTINEL, shallow=False):
        """Yields all nodes with associated values with given prefix.
        """
        node, _ = self._get_node(prefix)
        for path, value in node.iterate(list(self.__path_from_key(prefix)),
                                        shallow, self._iteritems):
            yield (self._key_from_path(path), value)

    def iterkeys(self, prefix=_SENTINEL, shallow=False):
        """Yields all keys having associated values with given prefix.
        """
        for key, _ in self.iteritems(prefix=prefix, shallow=shallow):
            yield key

    def itervalues(self, prefix=_SENTINEL, shallow=False):
        """Yields all values associated with keys with given prefix.
        """
        node, _ = self._get_node(prefix)
        for _, value in node.iterate(list(self.__path_from_key(prefix)),
                                     shallow, self._iteritems):
            yield value

    def items(self, prefix=_SENTINEL, shallow=False):
        """Returns a list of ``(key, value)`` pairs in given subtrie.
        """
        return list(self.iteritems(prefix=prefix, shallow=shallow))

    def keys(self, prefix=_SENTINEL, shallow=False):
        """Returns a list of all the keys, with given prefix, in the trie.
        """
        return list(self.iterkeys(prefix=prefix, shallow=shallow))

    def values(self, prefix=_SENTINEL, shallow=False):
        """Returns a list of values in given subtrie.
        """
        return list(self.itervalues(prefix=prefix, shallow=shallow))

    # pylint: enable=arguments-differ

    def __len__(self):
        """Returns number of values in a trie.
        """
        return sum(1 for _ in self.itervalues())

    def __nonzero__(self):
        return bool(self._root)

    HAS_VALUE = 1
    HAS_SUBTRIE = 2

    def has_node(self, key):
        """Returns whether given node is in the trie.
        """
        try:
            node, _ = self._get_node(key)
        except KeyError:
            return 0
        return ((self.HAS_VALUE * int(node.value is not _SENTINEL)) |
                (self.HAS_SUBTRIE * int(bool(node.children))))

    def has_key(self, key):
        """Indicates whether given key has value associated with it.
        """
        return bool(self.has_node(key) & self.HAS_VALUE)

    def has_subtrie(self, key):
        """Returns whether given key is a prefix of another key in the trie.
        """
        return bool(self.has_node(key) & self.HAS_SUBTRIE)

    @staticmethod
    def _slice_maybe(key_or_slice):
        """Checks whether argument is a slice or a plain key.
        """
        if isinstance(key_or_slice, slice):
            if key_or_slice.stop is not None or key_or_slice.step is not None:
                raise TypeError(key_or_slice)
            return key_or_slice.start, True
        return key_or_slice, False

    def __getitem__(self, key_or_slice):
        """Returns value associated with given key or raises KeyError.
        """
        if self._slice_maybe(key_or_slice)[1]:
            return self.itervalues(key_or_slice.start)
        node, _ = self._get_node(key_or_slice)
        if node.value is _SENTINEL:
            raise KeyError(key_or_slice)
        return node.value

    def _set(self, key, value, only_if_missing=False, clear_children=False):
        """Sets value for a given key.
        """
        node, _ = self._get_node(key, create=True)
        if not only_if_missing or node.value is _SENTINEL:
            node.value = value
        if clear_children:
            node.children.clear()
        return node.value

    def __setitem__(self, key_or_slice, value):
        """Sets value associated with given key.
        """
        key, is_slice = self._slice_maybe(key_or_slice)
        self._set(key, value, clear_children=is_slice)

    def setdefault(self, key, value=None):
        """Sets value of a given node if not set already.  Also returns it.
        """
        return self._set(key, value, only_if_missing=True)

    @staticmethod
    def _cleanup_trace(trace):
        """Removes empty nodes present on specified trace.

        Args:
            trace: Trace to the node to cleanup as returned by
                :func:`Trie._get_node`.
        """
        i = len(trace) - 1  # len(path) >= 1 since root is always there
        step, node = trace[i]
        while i and not node:
            i -= 1
            parent_step, parent = trace[i]
            del parent.children[step]
            step, node = parent_step, parent

    def _pop_from_node(self, node, trace, default=_SENTINEL):
        """Removes a value from given node.
        """
        if node.value is not _SENTINEL:
            value = node.value
            node.value = _SENTINEL
            self._cleanup_trace(trace)
            return value
        elif default is _SENTINEL:
            raise KeyError()
        else:
            return default

    def pop(self, key, default=_SENTINEL):
        """Deletes value associated with given key and returns it.
        """
        try:
            return self._pop_from_node(*self._get_node(key))
        except KeyError:
            if default is not _SENTINEL:
                return default
            raise

    def popitem(self):
        """Deletes an arbitrary value from the trie and returns it.
        """
        if not self:
            raise KeyError()
        node = self._root
        trace = [(None, node)]
        while node.value is _SENTINEL:
            step = next(_iterkeys(node.children))
            node = node.children[step]
            trace.append((step, node))
        return (self._key_from_path(
            (step for step, _ in trace[1:])), self._pop_from_node(node, trace))

    def __delitem__(self, key_or_slice):
        """Deletes value associated with given key or raises KeyError.
        """
        key, is_slice = self._slice_maybe(key_or_slice)
        node, trace = self._get_node(key)
        if is_slice:
            node.children.clear()
        elif node.value is _SENTINEL:
            raise KeyError(key)
        node.value = _SENTINEL
        self._cleanup_trace(trace)

    def prefixes(self, key):
        """Walks towards the node specified by key and yields all found items.
        """
        node = self._root
        path = self.__path_from_key(key)
        pos = 0
        while True:
            if node.value is not _SENTINEL:
                yield self._key_from_path(path[:pos]), node.value
            if pos == len(path):
                break
            node = node.children.get(path[pos])
            if not node:
                break
            pos += 1

    def __path_from_key(self, key):
        """Converts a user visible key object to internal path representation.
        """
        return () if key is _SENTINEL else key

    def _key_from_path(self, path):
        return ''.join(path)


###################################
# Text to NER
###################################
def word_dict_to_trie(word_dict):
    """Build CharTrie from dict
    :param word_dict:
    {
        "car_brand": [
            "localmotors", "mg", "mini", "nevs国能汽车",
            "上汽仪征", "上海汇众", "世爵", "东南", "东沃"],
        "car_type": ["mpv","suv","客车","工程车","新能源","皮卡"],
        "car_series": ["风光580"]
    }
    :return:
    """
    ctrie = CharTrie()
    for w_type, ws in word_dict.items():
        for w in ws:
            ctrie[w] = w_type
    return ctrie


def trie_ner(text, ctrie, lower=True):
    p_first = 0
    cur_str = ""
    res = []
    txt = text.lower() if lower else text
    for i in range(len(txt)):
        check_word = cur_str + txt[i]
        if ctrie.has_subtrie(check_word):
            if cur_str == "":
                p_first = i
            cur_str += txt[i]
        elif check_word in ctrie:
            res.append({
                "start": p_first,
                "end": i + 1,
                "value": check_word,
                "entity": ctrie.get(check_word)
            })
            cur_str = ""
        elif cur_str in ctrie:
            res.append({
                "start": p_first,
                "end": i,
                "value": cur_str,
                "entity": ctrie.get(cur_str)
            })
            cur_str = txt[i]
            if ctrie.has_subtrie(cur_str):
                p_first = i
        else:
            if ctrie.has_subtrie(txt[i]):
                p_first = i
                cur_str = txt[i]
            else:
                cur_str = ""

    return res


def read_scp(fn):
    with open(fn, 'r', encoding="utf-8") as fp:
        for line in fp:
            line_split = line.strip().split("\t", 1)
            if len(line_split) != 2:
                continue
            utt, text = line_split
            yield utt, text


def read_hotwords(fn):
    with open(fn, 'r', encoding='utf-8') as fp:
        hotwords = yaml.load(fp.read(), Loader=yaml.FullLoader)
    return list(hotwords.keys())


###################################
# NER F1
###################################
def cal_f1(ner_dict_pred, ner_dict_label):
    acc_dict = {"correct": 0, "predcount": 0}
    recall_dict = {"correct": 0, "labelcount": 0}
    for utt in ner_dict_pred:
        pred_ners = ner_dict_pred[utt]  # list
        label_ners = ner_dict_label[utt]  # list
        for w in pred_ners:
            acc_dict["predcount"] += 1
            if w in label_ners:
                acc_dict["correct"] += 1
            # else:
            #     print(f"{utt}")
        for w in label_ners:
            recall_dict["labelcount"] += 1
            if w in pred_ners:
                recall_dict["correct"] += 1

    acc = acc_dict["correct"] / acc_dict["predcount"]
    recall = recall_dict["correct"] / recall_dict["labelcount"]
    f1 = 2 * acc * recall / (acc + recall)

    return acc, recall, f1


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="data/aishell1_text_hotwords")
    parser.add_argument(
        "--preds",
        default="data/with_hotwords_ali.log;data/without_hotwords_ali.log")
    parser.add_argument("--hotword", default="data/hotwords.yaml")
    return parser.parse_args()


def text2ner(in_file, out_file, ctrie):
    ner_dict = {}
    with open(out_file, 'w', encoding="utf-8") as fpw:
        for utt, text in read_scp(in_file):
            res = trie_ner(text, ctrie)
            # res = [{'start': 19, 'end': 22, 'value': '邓郁松', 'entity': 'hotword'}]
            ners = ",".join([i['value'] for i in res])
            fpw.write(f"{utt}\t{text}\t{ners}\n")
            ner_dict[utt] = ners.split(",") if len(ners) > 0 else []
    return ner_dict


def main():
    args = get_args()
    label = args.label
    preds = args.preds.split(";")
    hotwords = read_hotwords(args.hotword)
    ctrie = word_dict_to_trie({"hotword": hotwords})

    # extract ner
    ner_dict_label = text2ner(label, label + ".ner", ctrie)
    for pred in preds:
        if not os.path.exists(pred):
            continue
        ner_dict_preds = text2ner(pred, pred + ".ner", ctrie)
        assert len(ner_dict_preds) == len(ner_dict_label)

        # compute f1
        wo_acc, wo_recall, wo_f1 = cal_f1(ner_dict_preds, ner_dict_label)
        print(f"[{pred}] acc={wo_acc}, recall={wo_recall}, f1={wo_f1}")


if __name__ == '__main__':
    main()
