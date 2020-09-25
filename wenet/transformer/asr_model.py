from typing import List, Optional, Tuple, Union
import logging

import torch
from typeguard import check_argument_types

from wenet.transformer.encoder import TransformerEncoder
from wenet.transformer.decoder import TransformerDecoder
from wenet.transformer.ctc import CTC
from wenet.transformer.label_smoothing_loss import LabelSmoothingLoss
from wenet.utils.common import add_sos_eos, th_accuracy
from wenet.utils.mask import subsequent_mask


class ASRModel(torch.nn.Module):
    """CTC-attention hybrid Encoder-Decoder model"""
    def __init__(
        self,
        vocab_size: int,
        encoder: TransformerEncoder,
        decoder: TransformerDecoder,
        ctc: CTC,
        ctc_weight: float = 0.5,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
    ):
        #assert check_argument_types()
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight

        self.encoder = encoder
        self.decoder = decoder
        self.ctc = ctc
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        encoder_out, encoder_out_lens, encoder_mask = self.encoder(
            speech, speech_lengths)

        # 2a. Attention-decoder branch
        loss_att, acc_att = self._calc_att_loss(encoder_out, encoder_mask,
                                                text, text_lengths)

        # 2b. CTC branch
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, text, text_lengths)

        if self.ctc_weight == 0.0:
            loss = loss_att
        elif self.ctc_weight == 1.0:
            loss = loss_ctc
        else:
            loss = self.ctc_weight * loss_ctc + (1 -
                                                 self.ctc_weight) * loss_att
        return loss, loss_att, loss_ctc

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

        # 1. Forward decoder
        decoder_out, _ = self.decoder(encoder_out, encoder_mask, ys_in_pad,
                                      ys_in_lens)

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )
        return loss_att, acc_att

    #def recognize(self,
    #              speech: torch.Tensor,
    #              speech_lengths: torch.Tensor,
    #              beam: int = 10,
    #              penalty: float = 0.0,
    #              maxlen: int = 0)
    #    assert speech.shape[0] == speech_lengths.shape[0]
    #    if maxlen == 0: maxlen = speech.shape[0]
    #    device = speech.device
    #    # 1. Encoder
    #    encoder_out, encoder_out_lens, encoder_mask = self.encoder(speech, speech_lengths)

    #    hyp = torch.tensor([self.sos for _ in range(beam)], dtype=torch.long)
    #    cache = None
    #    # 2. Decoder forward step by step
    #    for i in range(maxlen):
    #        mask = subsequent_mask(hyp.size(-1), device=.device).unsqueeze(0)
    #        logp, cache = self.decoder.forward_one_step(
    #            hyp, mask, encoder_out, cache)
    #        pass
