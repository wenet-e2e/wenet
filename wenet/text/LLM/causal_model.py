from typing import Dict, Optional
import torch
from wenet.text.LLM.decoder import DecoderOnly
from wenet.transformer.label_smoothing_loss import LabelSmoothingLoss
from wenet.utils.common import IGNORE_ID, add_sos_eos, th_accuracy
from wenet.utils.mask import make_non_pad_mask, subsequent_mask


class CausalLM(torch.nn.Module):

    def __init__(
        self,
        vocab_size: int,
        decoder: DecoderOnly,
        tie_word_embedding: bool = False,
        linear_bias: bool = False,
        special_tokens: Optional[dict] = None,
        ignore_id: int = IGNORE_ID,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
    ) -> None:
        super().__init__()

        self.embed = torch.nn.Embedding(vocab_size, decoder.hidden_size)
        self.out = torch.nn.Linear(decoder.hidden_size,
                                   vocab_size,
                                   bias=linear_bias)
        if tie_word_embedding:
            self.out.weight = self.embed.weight

        self.decoder = decoder
        self.sos = (vocab_size - 1 if special_tokens is None else
                    special_tokens.get("<sos>", vocab_size - 1))
        self.eos = (vocab_size - 1 if special_tokens is None else
                    special_tokens.get("<eos>", vocab_size - 1))

        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        self.ignore_id = ignore_id

    @torch.jit.ignore(drop=True)
    def forward(
        self,
        batch: dict,
        device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:

        text = batch['text'].to(device)
        text_length = batch['text_lengths'].to(device)

        ys_in_pad, ys_out_pad = add_sos_eos(text, self.sos, self.eos,
                                            self.ignore_id)
        ys_in_lens = text_length + 1

        # TODO: fix maxlength for pading to max
        tgt_mask = make_non_pad_mask(ys_in_lens).unsqueeze(1).to(
            text.device)  # (B, 1, L)
        causal_mask = subsequent_mask(tgt_mask.size(-1),
                                      device=tgt_mask.device).unsqueeze(
                                          0)  # (1,L,L)
        att_mask = causal_mask & tgt_mask  # (B, L, L)

        embeding = self.embed(ys_in_pad)
        decoder_out = self.out(self.decoder(embeding,
                                            att_mask))  # (B, L, vocab_size)

        loss = self.criterion_att(decoder_out, ys_out_pad)
        acc = th_accuracy(decoder_out.view(-1, self.vocab_size),
                          ys_out_pad,
                          ignore_label=self.ignore_id)

        # TODO: ppl
        return {"loss": loss, "ppl": None, "th_accuracy": acc}

    def generate(self):
        pass
