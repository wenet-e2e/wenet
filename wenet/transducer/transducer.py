from typing import Dict, List, Optional, Tuple, Union

import torch
import torchaudio
from torch import nn
from typeguard import check_argument_types
from wenet.transducer.joint import TransducerJoint
from wenet.transducer.predictor import RNNPredictor
from wenet.transformer.cmvn import GlobalCMVN
from wenet.transformer.ctc import CTC
from wenet.transformer.decoder import BiTransformerDecoder, TransformerDecoder
from wenet.transformer.encoder import ConformerEncoder, TransformerEncoder
from wenet.transformer.label_smoothing_loss import LabelSmoothingLoss
from wenet.utils.cmvn import load_cmvn
from wenet.utils.common import (IGNORE_ID, add_sos_eos, reverse_pad_list,
                                th_accuracy)


class Transducer(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        blank_id: int,
        encoder: nn.Module,
        predictor: nn.Module,
        joint: nn.Module,
        attention_decoder: Optional[Union[TransformerDecoder,
                                          BiTransformerDecoder]] = None,
        ctc: Optional[CTC] = None,
        ctc_weight: float = 0,
        ignore_id: int = IGNORE_ID,
        reverse_weight: float = 0.0,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        transducer_weight: float = 1.0,
        attention_weight: float = 0.0,
    ) -> None:
        assert check_argument_types()
        assert attention_weight + ctc_weight + transducer_weight == 1.0
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.joint = joint

        # NOTE(Mddct): in transducer sos eos blank_id should be same
        self.blank_id = blank_id
        self.sos = self.blank_id
        self.eos = self.blank_id
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id

        self.transducer_weight = transducer_weight
        self.attention_decoder = attention_decoder
        self.reverse_weight = reverse_weight
        self.ctc = ctc
        self.ctc_weight = ctc_weight
        self.attention_decoder_weight = 1 - self.transducer_weight - self.ctc_weight
        assert self.attention_decoder_weight >= 0
        if attention_decoder is not None:
            self.criterion_att = LabelSmoothingLoss(
                size=vocab_size,
                padding_idx=ignore_id,
                smoothing=lsm_weight,
                normalize_length=length_normalized_loss,
            )

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Frontend + Encoder + predictor + joint + loss

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

        # Encoder
        encoder_out, encoder_mask = self.encoder(speech, speech_lengths)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)

        loss_att: Optional[torch.Tensor] = None
        # optional attention decoder
        if self.transducer_weight != 1.0 and self.attention_decoder is not None:
            loss_att, _ = self._calc_att_loss(encoder_out, encoder_mask, text,
                                              text_lengths)
        loss_ctc: Optional[torch.Tensor] = None
        # optional ctc
        if self.ctc_weight != 0.0 and self.ctc is not None:
            loss_ctc = self.ctc(encoder_out, encoder_out_lens, text,
                                text_lengths)
        else:
            loss_ctc = None

        # predictor
        ys_in_pad, _ = add_sos_eos(text, self.sos, self.eos, self.ignore_id)
        predictor_out = self.predictor(ys_in_pad)

        # joint
        joint_out = self.joint(encoder_out, predictor_out)

        # NOTE(Mddct): some loss implementation require pad valid is zero
        # torch.int32 rnn_loss required
        text = text.to(torch.int64)
        text = torch.where(text == self.ignore_id, 0, text).to(torch.int32)
        text_lengths = text_lengths.to(torch.int32)
        encoder_out_lens = encoder_out_lens.to(torch.int32)
        loss = torchaudio.functional.rnnt_loss(joint_out,
                                               text,
                                               encoder_out_lens,
                                               text_lengths,
                                               blank=self.blank_id,
                                               reduction="mean")
        loss_rnnt = loss
        if loss_ctc is not None:
            loss = loss + self.ctc_weight * loss_ctc.sum()
        if loss_att is not None:
            loss = loss + self.attention_decoder_weight * loss_att.sum()
        # NOTE: 'loss' must be in dict
        return {
            'loss': loss,
            'loss_rnnt': loss_rnnt,
            'loss_att': loss_att,
            'loss_ctc': loss_ctc
        }

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
        decoder_out, r_decoder_out, _ = self.attention_decoder(
            encoder_out, encoder_mask, ys_in_pad, ys_in_lens, r_ys_in_pad,
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

    def greedy_search(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
    ) -> List[List[int]]:
        """ greedy search

        Args:
            speech (torch.Tensor): (batch=1, max_len, feat_dim)
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
        # TODO(Mddct): batch decode
        assert speech.size(0) == 1
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        batch_size = speech.shape[0]
        # Let's assume B = batch_size
        # NOTE(Mddct): only support non streamming for now
        num_decoding_left_chunks = -1
        decoding_chunk_size = -1
        simulate_streaming = False
        encoder_out, encoder_mask = self.encoder(
            speech,
            speech_lengths,
            decoding_chunk_size,
            num_decoding_left_chunks,
        )
        maxlen = encoder_out.size(1)
        encoder_out_lens = encoder_mask.squeeze(1).sum()

        # fake padding
        padding = torch.zeros(1, 1)
        # sos
        pred_input_step = torch.tensor([self.sos]).reshape(1, 1)
        state_m, state_c = self.predictor.init_state(1, method="zero")
        t = 0
        hyps = []
        prev_out_nblk = False
        pred_out_step = None
        per_frame_max_noblk = 5
        per_frame_noblk = 0
        while t < encoder_out_lens:
            encoder_out_step = encoder_out[:, t:t + 1, :]  # [1, 1, E]
            if not prev_out_nblk:
                pred_out_step, state_m, state_c = self.predictor.forward_step(
                    pred_input_step, padding, state_m, state_c)  # [1, 1, P]

            joint_out_step = self.joint(encoder_out_step,
                                        pred_out_step)  # [1,1,v]
            joint_out_probs = joint_out_step.log_softmax(dim=-1)
            joint_out_max = joint_out_probs.argmax(dim=-1).squeeze()  # []
            if joint_out_max != self.blank_id:
                hyps.append(joint_out_max)
                prev_out_nblk = True
                per_frame_noblk = per_frame_noblk + 1

            if joint_out_max == self.blank_id or per_frame_noblk > per_frame_max_noblk:
                prev_out_nblk = False
                # TODO(Mddct): make t in chunk for streamming
                # or t should't be too lang to predict none blank
                t = t + 1
                per_frame_noblk = 0

        return [hyps]

    @torch.jit.export
    def forward_encoder_chunk(
        self,
        xs: torch.Tensor,
        offset: int,
        required_cache_size: int,
        att_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
        cnn_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        return self.encoder.forward_chunk(xs, offset, required_cache_size,
                                          att_cache, cnn_cache)

    @torch.jit.export
    def forward_predictor_step(
        self, xs: torch.Tensor, state_m: torch.Tensor, state_c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # fake padding
        padding = torch.zeros(1, 1)
        return self.predictor.forward_step(xs, padding, state_m, state_c)

    @torch.jit.export
    def forward_joint_step(self, enc_out: torch.Tensor,
                           pred_out: torch.Tensor) -> torch.Tensor:
        return self.joint(enc_out, pred_out)

    @torch.jit.export
    def forward_predictor_init_state(
            self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.predictor.init_state(1)


def init_transducer_asr_model(configs):
    assert "blank_id" in configs
    if configs['cmvn_file'] is not None:
        mean, istd = load_cmvn(configs['cmvn_file'], configs['is_json_cmvn'])
        global_cmvn = GlobalCMVN(
            torch.from_numpy(mean).float(),
            torch.from_numpy(istd).float())
    else:
        global_cmvn = None

    input_dim = configs['input_dim']
    vocab_size = configs['vocab_size']

    encoder_type = configs.get('encoder', 'conformer')
    predictor_type = configs.get('predictor', 'rnn')

    if encoder_type == 'conformer':
        encoder = ConformerEncoder(input_dim,
                                   global_cmvn=global_cmvn,
                                   **configs['encoder_conf'])
    else:
        encoder = TransformerEncoder(input_dim,
                                     global_cmvn=global_cmvn,
                                     **configs['encoder_conf'])
    if predictor_type == 'rnn':
        predictor = RNNPredictor(vocab_size, **configs['predictor_conf'])
    else:
        raise NotImplementedError("only rnn type support now")

    configs['joint_conf']['enc_output_size'] = configs['encoder_conf'][
        'output_size']
    configs['joint_conf']['pred_output_size'] = configs['predictor_conf'][
        'output_size']
    joint = TransducerJoint(vocab_size, **configs['joint_conf'])

    # optional attention decoder
    decoder_type = configs.get('decoder', '')
    if decoder_type == 'transformer':
        decoder = TransformerDecoder(vocab_size, encoder.output_size(),
                                     **configs['decoder_conf'])
    elif decoder_type == 'bitransformer':
        assert 0.0 < configs['model_conf']['reverse_weight'] < 1.0
        assert configs['decoder_conf']['r_num_blocks'] > 0
        decoder = BiTransformerDecoder(vocab_size, encoder.output_size(),
                                       **configs['decoder_conf'])
    else:
        decoder = None

    model = Transducer(vocab_size=vocab_size,
                       blank_id=configs["blank_id"],
                       predictor=predictor,
                       encoder=encoder,
                       attention_decoder=decoder,
                       joint=joint,
                       **configs['model_conf'])
    return model
