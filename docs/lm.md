# LM for WeNet

WeNet use n-gram based statistical lanuage model and WFST framework to support custom language model.

## Motivation

Why n-gram? This may be the first question many people will ask.
Now that LM based on RNN and Transformer is in full swing, why does WeNet go backwards?
The reason is simple. it is for productivity.
The n-gram-based language model has mature and complete training tools,
any amount of corpus can be trained, the training is very fast, hotfix is easy,
and it has a wide range of mature applications in actual products.

Why WFST? It may be the second question many people will ask.
Since both industry and research have been working so hard to abandon traditional speech recognition,
especially the complex decoding technology. Why WeNet goes back?
The reason is also very simple, it is for productivity.
WFST is a standard and powerful tool in traditional speech recognition.
And based on this solution, we have mature and complete bug fix solutions and product solutions,
such as that we can use the replace function in WFST for class-based personalization such as contact recognition.

Therefore, just like WeNet's design goal "Production first and Production Ready",
LM support in WeNet also puts productivity as the first priority.
So it draws on many very productive tools and solutions accumulated in traditional speech recognition.
The difference to traditional speech recognition are:

1. The training in WeNet is pure end to end.
2. As described below, LM is optional in decoding, you can choose whether to use LM according to your needs and application scenarios.


## System Design

The whole system is shown in the bellowing picture. There are two ways to generate N-best.

![LM System Design](./images/lm_system.png)

1. Without LM, we use CTC prefix beam search to generate N-best.
2. With LM, we use CTC WFST search to generate N-best and CTC WFST search is the traditional WFST based decoder.

There are two main parts of CTC WFST based search.

The first is building the decoding graph, which is to compose the model unit T, lexicon L and language model G into one unified graph TLG. And in which:
1. T is the model unit in E2E training. Typically it's char in Chinese, char or BPE in english.
2. L is lexicon, the lexicon is very simple. What we need to do is just split a word into its modeling unit sequence.
For example, the word "我们" is split into two chars "我 们", and the word "APPLE" is split into five letters "A P P L E" .
We can see there is no phonemes and there is no need design pronunciation on purpose.
3. G is the language model, namely compling the n-gram to standard WFST representation.

The second is decoder, which is same to the traditional decoder, which use standard viterbi beam search algorithm in decoding.

## Implementation

WeNet draws on the decoder and related tools in Kaldi to support LM and WFST based decocding.
For ease of use and keep independence, we directly migrated the code related to decoding in Kaldi to [this directory](https://github.com/mobvoi/wenet/tree/main/runtime/core/kaldi) in WeNet runtime.
And modify and organize according to the following principles:
1. In order to minimize changes, the migrated code remains the same directory structure as the original.
2. We use GLOG to replace the log system in Kaldi.
3. We modify the code format to meet the lint requirements of the code style in WeNet.

The core code is https://github.com/mobvoi/wenet/blob/main/runtime/core/decoder/ctc_wfst_beam_search.cc, which wraps the LatticeFasterDecoder in Kaldi. And we use blank frame skipping to speed up decoding.

In addition, WeNet also migrated related tools for building the decoding graph,
such as arpa2fst, fstdeterminizestar, fsttablecompose, fstminimizeencoded and other tools.
So all the tools related to LM are built-in tools, and can be used out of box.


## Experiments

We get consistent gain(3%~10%) on different datasets, including aishell, aishell2 and librispeech, please go to the corresponding example dataset for the details. 
