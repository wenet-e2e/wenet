# Copyright (c) 2023 Binbin Zhang (binbzha@qq.com)
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

from typing import Dict, List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

from wenet.transformer.asr_model import ASRModel
from wenet.transformer.ctc import CTC
from wenet.transformer.decoder import TransformerDecoder
from wenet.transformer.encoder import TransformerEncoder
from wenet.utils.common import (IGNORE_ID, add_sos_eos, reverse_pad_list)


class K2Model(ASRModel):

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
            special_tokens: dict = None,
            device: torch.device = torch.device("cuda"),
    ):
        super().__init__(vocab_size,
                         encoder,
                         decoder,
                         ctc,
                         ctc_weight,
                         ignore_id,
                         reverse_weight,
                         lsm_weight,
                         length_normalized_loss,
                         special_tokens=special_tokens)
        self.lfmmi_dir = lfmmi_dir
        self.device = device
        if self.lfmmi_dir != '':
            self.load_lfmmi_resource()

    @torch.jit.unused
    def _forward_ctc(
            self, encoder_out: torch.Tensor, encoder_mask: torch.Tensor,
            text: torch.Tensor,
            text_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        loss_ctc, ctc_probs = self._calc_lfmmi_loss(encoder_out, encoder_mask,
                                                    text)
        return loss_ctc, ctc_probs

    @torch.jit.unused
    def load_lfmmi_resource(self):
        try:
            import icefall
        except ImportError:
            print('Error: Failed to import icefall')
        with open('{}/tokens.txt'.format(self.lfmmi_dir), 'r') as fin:
            for line in fin:
                arr = line.strip().split()
                if arr[0] == '<sos/eos>':
                    self.sos_eos_id = int(arr[1])
        device = torch.device(self.device)
        self.graph_compiler = icefall.mmi_graph_compiler.MmiTrainingGraphCompiler(
            self.lfmmi_dir,
            device=device,
            oov="<UNK>",
            sos_id=self.sos_eos_id,
            eos_id=self.sos_eos_id,
        )
        self.lfmmi = icefall.mmi.LFMMILoss(
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

    @torch.jit.unused
    def _calc_lfmmi_loss(self, encoder_out, encoder_mask, text):
        try:
            import k2
        except ImportError:
            print('Error: Failed to import k2')
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
        return loss, ctc_probs

    def load_hlg_resource_if_necessary(self, hlg, word):
        try:
            import k2
        except ImportError:
            print('Error: Failed to import k2')
        if not hasattr(self, 'hlg'):
            device = torch.device(self.device)
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
        try:
            import icefall
        except ImportError:
            print('Error: Failed to import icefall')
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
        lattice = icefall.decode.get_lattice(
            nnet_output=ctc_probs,
            decoding_graph=self.hlg,
            supervision_segments=supervision_segments,
            search_beam=20,
            output_beam=7,
            min_active_states=30,
            max_active_states=10000,
            subsampling_factor=4)
        best_path = icefall.decode.one_best_decoding(lattice=lattice,
                                                     use_double_scores=True)
        hyps = icefall.utils.get_texts(best_path)
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
        try:
            import k2
            import icefall
        except ImportError:
            print('Error: Failed to import k2 & icefall')
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
        lattice = icefall.decode.get_lattice(
            nnet_output=ctc_probs,
            decoding_graph=self.hlg,
            supervision_segments=supervision_segments,
            search_beam=20,
            output_beam=7,
            min_active_states=30,
            max_active_states=10000,
            subsampling_factor=4)
        nbest = icefall.decode.Nbest.from_lattice(
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
        hyps = icefall.utils.get_texts(best_path)
        hyps = [[symbol_table[k] for j in i for k in self.word_table[j]]
                for i in hyps]
        return hyps
