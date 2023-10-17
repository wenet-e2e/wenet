# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Di Wu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from ESPnet(https://github.com/espnet/espnet)

from typing import Dict, List, Optional, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

try:
    import k2
    from icefall.utils import get_texts
    from icefall.decode import get_lattice, Nbest, one_best_decoding
    from icefall.mmi import LFMMILoss
    from icefall.mmi_graph_compiler import MmiTrainingGraphCompiler
except ImportError:
    print('Warning: Failed to import k2 & icefall, which are for LF-MMI/hlg')

from wenet.transformer.ctc import CTC
from wenet.transformer.decoder import TransformerDecoder
from wenet.transformer.encoder import TransformerEncoder
from wenet.transformer.label_smoothing_loss import LabelSmoothingLoss
from wenet.transformer.search import (ctc_greedy_search,
                                      ctc_prefix_beam_search,
                                      attention_beam_search,
                                      attention_rescoring)
from wenet.utils.common import (IGNORE_ID,
                                add_sos_eos,
                                th_accuracy,
                                reverse_pad_list)
from wenet.utils.context_graph import ContextGraph


class ASRModel(torch.nn.Module):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        vocab_size: int,
        encoder: TransformerEncoder,
        decoder: TransformerDecoder,
        ctc: CTC,
        ctc_weight: float = 0.5,
        ignore_id: int = IGNORE_ID,
        reverse_weight: float = 0.0,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        lfmmi_dir: str = '',
    ):
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.reverse_weight = reverse_weight

        self.encoder = encoder
        self.decoder = decoder
        self.ctc = ctc
        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )
        self.lfmmi_dir = lfmmi_dir
        if self.lfmmi_dir != '':
            self.load_lfmmi_resource()

    @torch.jit.ignore(drop=True)
    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        """

        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (speech.shape[0] == speech_lengths.shape[0] == text.shape[0] ==
                text_lengths.shape[0]), (speech.shape, speech_lengths.shape,
                                         text.shape, text_lengths.shape)
        # 1. Encoder
        encoder_out, encoder_mask = self.encoder(speech, speech_lengths)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)

        # 2a. Attention-decoder branch
        if self.ctc_weight != 1.0:
            loss_att, acc_att = self._calc_att_loss(encoder_out, encoder_mask,
                                                    text, text_lengths)
        else:
            loss_att = None

        # 2b. CTC branch or LF-MMI loss
        if self.ctc_weight != 0.0:
            if self.lfmmi_dir != '':
                loss_ctc = self._calc_lfmmi_loss(encoder_out, encoder_mask,
                                                 text)
            else:
                loss_ctc = self.ctc(encoder_out, encoder_out_lens, text,
                                    text_lengths)
        else:
            loss_ctc = None

        if loss_ctc is None:
            loss = loss_att
        elif loss_att is None:
            loss = loss_ctc
        else:
            loss = self.ctc_weight * loss_ctc + (1 -
                                                 self.ctc_weight) * loss_att
        return {"loss": loss, "loss_att": loss_att, "loss_ctc": loss_ctc}

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos,
                                            self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # reverse the seq, used for right to left decoder
        r_ys_pad = reverse_pad_list(ys_pad, ys_pad_lens, float(self.ignore_id))
        r_ys_in_pad, r_ys_out_pad = add_sos_eos(r_ys_pad, self.sos, self.eos,
                                                self.ignore_id)
        # 1. Forward decoder
        decoder_out, r_decoder_out, _ = self.decoder(encoder_out, encoder_mask,
                                                     ys_in_pad, ys_in_lens,
                                                     r_ys_in_pad,
                                                     self.reverse_weight)
        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        r_loss_att = torch.tensor(0.0)
        if self.reverse_weight > 0.0:
            r_loss_att = self.criterion_att(r_decoder_out, r_ys_out_pad)
        loss_att = loss_att * (
            1 - self.reverse_weight) + r_loss_att * self.reverse_weight
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )
        return loss_att, acc_att

    def _forward_encoder(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Let's assume B = batch_size
        # 1. Encoder
        if simulate_streaming and decoding_chunk_size > 0:
            encoder_out, encoder_mask = self.encoder.forward_chunk_by_chunk(
                speech,
                decoding_chunk_size=decoding_chunk_size,
                num_decoding_left_chunks=num_decoding_left_chunks
            )  # (B, maxlen, encoder_dim)
        else:
            encoder_out, encoder_mask = self.encoder(
                speech,
                speech_lengths,
                decoding_chunk_size=decoding_chunk_size,
                num_decoding_left_chunks=num_decoding_left_chunks
            )  # (B, maxlen, encoder_dim)
        return encoder_out, encoder_mask

    def recognize(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        beam_size: int = 10,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
    ) -> torch.Tensor:
        """ Apply beam search on attention decoder

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion

        Returns:
            torch.Tensor: decoding result, (batch, max_result_len)
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        # Let's assume B = batch_size and N = beam_size
        # 1. Encoder
        encoder_out, encoder_mask = self._forward_encoder(
            speech, speech_lengths, decoding_chunk_size,
            num_decoding_left_chunks,
            simulate_streaming)  # (B, maxlen, encoder_dim)
        return attention_beam_search(self, encoder_out, encoder_mask,
                                     beam_size)

    def ctc_greedy_search(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
    ) -> List[List[int]]:
        """ Apply CTC greedy search

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion
        Returns:
            List[List[int]]: best path result
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        batch_size = speech.shape[0]
        # Let's assume B = batch_size
        encoder_out, encoder_mask = self._forward_encoder(
            speech, speech_lengths, decoding_chunk_size,
            num_decoding_left_chunks,
            simulate_streaming)  # (B, maxlen, encoder_dim)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        ctc_probs = self.ctc.log_softmax(
            encoder_out)  # (B, maxlen, vocab_size)
        return ctc_greedy_search(ctc_probs, encoder_out_lens)

    def _ctc_prefix_beam_search(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        beam_size: int,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
        context_graph: ContextGraph = None,
    ) -> Tuple[List[List[int]], torch.Tensor]:
        """ CTC prefix beam search inner implementation

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion

        Returns:
            List[List[int]]: nbest results
            torch.Tensor: encoder output, (1, max_len, encoder_dim),
                it will be used for rescoring in attention rescoring mode
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        # Let's assume B = batch_size and N = beam_size
        # 1. Encoder forward and get CTC score
        encoder_out, encoder_mask = self._forward_encoder(
            speech, speech_lengths, decoding_chunk_size,
            num_decoding_left_chunks,
            simulate_streaming)  # (B, maxlen, encoder_dim)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        ctc_probs = self.ctc.log_softmax(
            encoder_out)  # (1, maxlen, vocab_size)
        return ctc_prefix_beam_search(ctc_probs, encoder_out_lens,
                                      beam_size), encoder_out

    def ctc_prefix_beam_search(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        beam_size: int,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
        context_graph: ContextGraph = None,
    ) -> List[int]:
        """ Apply CTC prefix beam search

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion

        Returns:
            List[int]: CTC prefix beam search nbest results
        """
        hyps, _ = self._ctc_prefix_beam_search(speech, speech_lengths,
                                               beam_size, decoding_chunk_size,
                                               num_decoding_left_chunks,
                                               simulate_streaming,
                                               context_graph)
        return hyps[0]

    def attention_rescoring(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        beam_size: int,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        ctc_weight: float = 0.0,
        simulate_streaming: bool = False,
        reverse_weight: float = 0.0,
        context_graph: ContextGraph = None,
    ) -> List[int]:
        """ Apply attention rescoring decoding, CTC prefix beam search
            is applied first to get nbest, then we resoring the nbest on
            attention decoder with corresponding encoder out

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion
            reverse_weight (float): right to left decoder weight
            ctc_weight (float): ctc score weight

        Returns:
            List[int]: Attention rescoring result
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        if reverse_weight > 0.0:
            # decoder should be a bitransformer decoder if reverse_weight > 0.0
            assert hasattr(self.decoder, 'right_decoder')
        device = speech.device
        batch_size = speech.shape[0]
        # For attention rescoring we only support batch_size=1
        assert batch_size == 1
        # encoder_out: (1, maxlen, encoder_dim), len(hyps) = beam_size
        hyps, encoder_out = self._ctc_prefix_beam_search(
            speech, speech_lengths, beam_size, decoding_chunk_size,
            num_decoding_left_chunks, simulate_streaming, context_graph)

        return attention_rescoring(self, hyps, encoder_out, ctc_weight, reverse_weight)


    @torch.jit.ignore(drop=True)
    def load_lfmmi_resource(self):
        with open('{}/tokens.txt'.format(self.lfmmi_dir), 'r') as fin:
            for line in fin:
                arr = line.strip().split()
                if arr[0] == '<sos/eos>':
                    self.sos_eos_id = int(arr[1])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.graph_compiler = MmiTrainingGraphCompiler(
            self.lfmmi_dir,
            device=device,
            oov="<UNK>",
            sos_id=self.sos_eos_id,
            eos_id=self.sos_eos_id,
        )
        self.lfmmi = LFMMILoss(
            graph_compiler=self.graph_compiler,
            den_scale=1,
            use_pruned_intersect=False,
        )
        self.word_table = {}
        with open('{}/words.txt'.format(self.lfmmi_dir), 'r') as fin:
            for line in fin:
                arr = line.strip().split()
                assert len(arr) == 2
                self.word_table[int(arr[1])] = arr[0]

    @torch.jit.ignore(drop=True)
    def _calc_lfmmi_loss(self, encoder_out, encoder_mask, text):
        ctc_probs = self.ctc.log_softmax(encoder_out)
        supervision_segments = torch.stack((
            torch.arange(len(encoder_mask)),
            torch.zeros(len(encoder_mask)),
            encoder_mask.squeeze(dim=1).sum(dim=1).to('cpu'),
        ), 1).to(torch.int32)
        dense_fsa_vec = k2.DenseFsaVec(
            ctc_probs,
            supervision_segments,
            allow_truncate=3,
        )
        text = [
            ' '.join([self.word_table[j.item()] for j in i if j != -1])
            for i in text
        ]
        loss = self.lfmmi(dense_fsa_vec=dense_fsa_vec, texts=text) / len(text)
        return loss

    def load_hlg_resource_if_necessary(self, hlg, word):
        if not hasattr(self, 'hlg'):
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
            self.hlg = k2.Fsa.from_dict(torch.load(hlg, map_location=device))
        if not hasattr(self.hlg, "lm_scores"):
            self.hlg.lm_scores = self.hlg.scores.clone()
        if not hasattr(self, 'word_table'):
            self.word_table = {}
            with open(word, 'r') as fin:
                for line in fin:
                    arr = line.strip().split()
                    assert len(arr) == 2
                    self.word_table[int(arr[1])] = arr[0]

    @torch.no_grad()
    def hlg_onebest(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
        hlg: str = '',
        word: str = '',
        symbol_table: Dict[str, int] = None,
    ) -> List[int]:
        self.load_hlg_resource_if_necessary(hlg, word)
        encoder_out, encoder_mask = self._forward_encoder(
            speech, speech_lengths, decoding_chunk_size,
            num_decoding_left_chunks,
            simulate_streaming)  # (B, maxlen, encoder_dim)
        ctc_probs = self.ctc.log_softmax(
            encoder_out)  # (1, maxlen, vocab_size)
        supervision_segments = torch.stack(
            (torch.arange(len(encoder_mask)), torch.zeros(len(encoder_mask)),
             encoder_mask.squeeze(dim=1).sum(dim=1).cpu()),
            1,
        ).to(torch.int32)
        lattice = get_lattice(nnet_output=ctc_probs,
                              decoding_graph=self.hlg,
                              supervision_segments=supervision_segments,
                              search_beam=20,
                              output_beam=7,
                              min_active_states=30,
                              max_active_states=10000,
                              subsampling_factor=4)
        best_path = one_best_decoding(lattice=lattice, use_double_scores=True)
        hyps = get_texts(best_path)
        hyps = [[symbol_table[k] for j in i for k in self.word_table[j]]
                for i in hyps]
        return hyps

    @torch.no_grad()
    def hlg_rescore(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
        lm_scale: float = 0,
        decoder_scale: float = 0,
        r_decoder_scale: float = 0,
        hlg: str = '',
        word: str = '',
        symbol_table: Dict[str, int] = None,
    ) -> List[int]:
        self.load_hlg_resource_if_necessary(hlg, word)
        device = speech.device
        encoder_out, encoder_mask = self._forward_encoder(
            speech, speech_lengths, decoding_chunk_size,
            num_decoding_left_chunks,
            simulate_streaming)  # (B, maxlen, encoder_dim)
        ctc_probs = self.ctc.log_softmax(
            encoder_out)  # (1, maxlen, vocab_size)
        supervision_segments = torch.stack(
            (torch.arange(len(encoder_mask)), torch.zeros(len(encoder_mask)),
             encoder_mask.squeeze(dim=1).sum(dim=1).cpu()),
            1,
        ).to(torch.int32)
        lattice = get_lattice(nnet_output=ctc_probs,
                              decoding_graph=self.hlg,
                              supervision_segments=supervision_segments,
                              search_beam=20,
                              output_beam=7,
                              min_active_states=30,
                              max_active_states=10000,
                              subsampling_factor=4)
        nbest = Nbest.from_lattice(
            lattice=lattice,
            num_paths=100,
            use_double_scores=True,
            nbest_scale=0.5,
        )
        nbest = nbest.intersect(lattice)
        assert hasattr(nbest.fsa, "lm_scores")
        assert hasattr(nbest.fsa, "tokens")
        assert isinstance(nbest.fsa.tokens, torch.Tensor)

        tokens_shape = nbest.fsa.arcs.shape().remove_axis(1)
        tokens = k2.RaggedTensor(tokens_shape, nbest.fsa.tokens)
        tokens = tokens.remove_values_leq(0)
        hyps = tokens.tolist()

        # cal attention_score
        hyps_pad = pad_sequence([
            torch.tensor(hyp, device=device, dtype=torch.long) for hyp in hyps
        ], True, self.ignore_id)  # (beam_size, max_hyps_len)
        ori_hyps_pad = hyps_pad
        hyps_lens = torch.tensor([len(hyp) for hyp in hyps],
                                 device=device,
                                 dtype=torch.long)  # (beam_size,)
        hyps_pad, _ = add_sos_eos(hyps_pad, self.sos, self.eos, self.ignore_id)
        hyps_lens = hyps_lens + 1  # Add <sos> at begining
        encoder_out_repeat = []
        tot_scores = nbest.tot_scores()
        repeats = [tot_scores[i].shape[0] for i in range(tot_scores.dim0)]
        for i in range(len(encoder_out)):
            encoder_out_repeat.append(encoder_out[i:i + 1].repeat(
                repeats[i], 1, 1))
        encoder_out = torch.concat(encoder_out_repeat, dim=0)
        encoder_mask = torch.ones(encoder_out.size(0),
                                  1,
                                  encoder_out.size(1),
                                  dtype=torch.bool,
                                  device=device)
        # used for right to left decoder
        r_hyps_pad = reverse_pad_list(ori_hyps_pad, hyps_lens, self.ignore_id)
        r_hyps_pad, _ = add_sos_eos(r_hyps_pad, self.sos, self.eos,
                                    self.ignore_id)
        reverse_weight = 0.5
        decoder_out, r_decoder_out, _ = self.decoder(
            encoder_out, encoder_mask, hyps_pad, hyps_lens, r_hyps_pad,
            reverse_weight)  # (beam_size, max_hyps_len, vocab_size)
        decoder_out = torch.nn.functional.log_softmax(decoder_out, dim=-1)
        decoder_out = decoder_out
        # r_decoder_out will be 0.0, if reverse_weight is 0.0 or decoder is a
        # conventional transformer decoder.
        r_decoder_out = torch.nn.functional.log_softmax(r_decoder_out, dim=-1)
        r_decoder_out = r_decoder_out

        decoder_scores = torch.tensor([
            sum([decoder_out[i, j, hyps[i][j]] for j in range(len(hyps[i]))])
            for i in range(len(hyps))
        ],
                                      device=device)  # noqa
        r_decoder_scores = []
        for i in range(len(hyps)):
            score = 0
            for j in range(len(hyps[i])):
                score += r_decoder_out[i, len(hyps[i]) - j - 1, hyps[i][j]]
            score += r_decoder_out[i, len(hyps[i]), self.eos]
            r_decoder_scores.append(score)
        r_decoder_scores = torch.tensor(r_decoder_scores, device=device)

        am_scores = nbest.compute_am_scores()
        ngram_lm_scores = nbest.compute_lm_scores()
        tot_scores = am_scores.values + lm_scale * ngram_lm_scores.values + \
            decoder_scale * decoder_scores + r_decoder_scale * r_decoder_scores
        ragged_tot_scores = k2.RaggedTensor(nbest.shape, tot_scores)
        max_indexes = ragged_tot_scores.argmax()
        best_path = k2.index_fsa(nbest.fsa, max_indexes)
        hyps = get_texts(best_path)
        hyps = [[symbol_table[k] for j in i for k in self.word_table[j]]
                for i in hyps]
        return hyps

    @torch.jit.export
    def subsampling_rate(self) -> int:
        """ Export interface for c++ call, return subsampling_rate of the
            model
        """
        return self.encoder.embed.subsampling_rate

    @torch.jit.export
    def right_context(self) -> int:
        """ Export interface for c++ call, return right_context of the model
        """
        return self.encoder.embed.right_context

    @torch.jit.export
    def sos_symbol(self) -> int:
        """ Export interface for c++ call, return sos symbol id of the model
        """
        return self.sos

    @torch.jit.export
    def eos_symbol(self) -> int:
        """ Export interface for c++ call, return eos symbol id of the model
        """
        return self.eos

    @torch.jit.export
    def forward_encoder_chunk(
        self,
        xs: torch.Tensor,
        offset: int,
        required_cache_size: int,
        att_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
        cnn_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Export interface for c++ call, give input chunk xs, and return
            output from time 0 to current chunk.

        Args:
            xs (torch.Tensor): chunk input, with shape (b=1, time, mel-dim),
                where `time == (chunk_size - 1) * subsample_rate + \
                        subsample.right_context + 1`
            offset (int): current offset in encoder output time stamp
            required_cache_size (int): cache size required for next chunk
                compuation
                >=0: actual cache size
                <0: means all history cache is required
            att_cache (torch.Tensor): cache tensor for KEY & VALUE in
                transformer/conformer attention, with shape
                (elayers, head, cache_t1, d_k * 2), where
                `head * d_k == hidden-dim` and
                `cache_t1 == chunk_size * num_decoding_left_chunks`.
            cnn_cache (torch.Tensor): cache tensor for cnn_module in conformer,
                (elayers, b=1, hidden-dim, cache_t2), where
                `cache_t2 == cnn.lorder - 1`

        Returns:
            torch.Tensor: output of current input xs,
                with shape (b=1, chunk_size, hidden-dim).
            torch.Tensor: new attention cache required for next chunk, with
                dynamic shape (elayers, head, ?, d_k * 2)
                depending on required_cache_size.
            torch.Tensor: new conformer cnn cache required for next chunk, with
                same shape as the original cnn_cache.

        """
        return self.encoder.forward_chunk(xs, offset, required_cache_size,
                                          att_cache, cnn_cache)

    @torch.jit.export
    def ctc_activation(self, xs: torch.Tensor) -> torch.Tensor:
        """ Export interface for c++ call, apply linear transform and log
            softmax before ctc
        Args:
            xs (torch.Tensor): encoder output

        Returns:
            torch.Tensor: activation before ctc

        """
        return self.ctc.log_softmax(xs)

    @torch.jit.export
    def is_bidirectional_decoder(self) -> bool:
        """
        Returns:
            torch.Tensor: decoder output
        """
        if hasattr(self.decoder, 'right_decoder'):
            return True
        else:
            return False

    @torch.jit.export
    def forward_attention_decoder(
        self,
        hyps: torch.Tensor,
        hyps_lens: torch.Tensor,
        encoder_out: torch.Tensor,
        reverse_weight: float = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Export interface for c++ call, forward decoder with multiple
            hypothesis from ctc prefix beam search and one encoder output
        Args:
            hyps (torch.Tensor): hyps from ctc prefix beam search, already
                pad sos at the begining
            hyps_lens (torch.Tensor): length of each hyp in hyps
            encoder_out (torch.Tensor): corresponding encoder output
            r_hyps (torch.Tensor): hyps from ctc prefix beam search, already
                pad eos at the begining which is used fo right to left decoder
            reverse_weight: used for verfing whether used right to left decoder,
            > 0 will use.

        Returns:
            torch.Tensor: decoder output
        """
        assert encoder_out.size(0) == 1
        num_hyps = hyps.size(0)
        assert hyps_lens.size(0) == num_hyps
        encoder_out = encoder_out.repeat(num_hyps, 1, 1)
        encoder_mask = torch.ones(num_hyps,
                                  1,
                                  encoder_out.size(1),
                                  dtype=torch.bool,
                                  device=encoder_out.device)

        # input for right to left decoder
        # this hyps_lens has count <sos> token, we need minus it.
        r_hyps_lens = hyps_lens - 1
        # this hyps has included <sos> token, so it should be
        # convert the original hyps.
        r_hyps = hyps[:, 1:]
        #   >>> r_hyps
        #   >>> tensor([[ 1,  2,  3],
        #   >>>         [ 9,  8,  4],
        #   >>>         [ 2, -1, -1]])
        #   >>> r_hyps_lens
        #   >>> tensor([3, 3, 1])

        # NOTE(Mddct): `pad_sequence` is not supported by ONNX, it is used
        #   in `reverse_pad_list` thus we have to refine the below code.
        #   Issue: https://github.com/wenet-e2e/wenet/issues/1113
        # Equal to:
        #   >>> r_hyps = reverse_pad_list(r_hyps, r_hyps_lens, float(self.ignore_id))
        #   >>> r_hyps, _ = add_sos_eos(r_hyps, self.sos, self.eos, self.ignore_id)
        max_len = torch.max(r_hyps_lens)
        index_range = torch.arange(0, max_len, 1).to(encoder_out.device)
        seq_len_expand = r_hyps_lens.unsqueeze(1)
        seq_mask = seq_len_expand > index_range  # (beam, max_len)
        #   >>> seq_mask
        #   >>> tensor([[ True,  True,  True],
        #   >>>         [ True,  True,  True],
        #   >>>         [ True, False, False]])
        index = (seq_len_expand - 1) - index_range  # (beam, max_len)
        #   >>> index
        #   >>> tensor([[ 2,  1,  0],
        #   >>>         [ 2,  1,  0],
        #   >>>         [ 0, -1, -2]])
        index = index * seq_mask
        #   >>> index
        #   >>> tensor([[2, 1, 0],
        #   >>>         [2, 1, 0],
        #   >>>         [0, 0, 0]])
        r_hyps = torch.gather(r_hyps, 1, index)
        #   >>> r_hyps
        #   >>> tensor([[3, 2, 1],
        #   >>>         [4, 8, 9],
        #   >>>         [2, 2, 2]])
        r_hyps = torch.where(seq_mask, r_hyps, self.eos)
        #   >>> r_hyps
        #   >>> tensor([[3, 2, 1],
        #   >>>         [4, 8, 9],
        #   >>>         [2, eos, eos]])
        r_hyps = torch.cat([hyps[:, 0:1], r_hyps], dim=1)
        #   >>> r_hyps
        #   >>> tensor([[sos, 3, 2, 1],
        #   >>>         [sos, 4, 8, 9],
        #   >>>         [sos, 2, eos, eos]])

        decoder_out, r_decoder_out, _ = self.decoder(
            encoder_out, encoder_mask, hyps, hyps_lens, r_hyps,
            reverse_weight)  # (num_hyps, max_hyps_len, vocab_size)
        decoder_out = torch.nn.functional.log_softmax(decoder_out, dim=-1)

        # right to left decoder may be not used during decoding process,
        # which depends on reverse_weight param.
        # r_dccoder_out will be 0.0, if reverse_weight is 0.0
        r_decoder_out = torch.nn.functional.log_softmax(r_decoder_out, dim=-1)
        return decoder_out, r_decoder_out
