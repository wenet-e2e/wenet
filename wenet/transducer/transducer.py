from typing import List, Optional, Tuple

import torch
import torchaudio
from torch import nn
from typeguard import check_argument_types
from wenet.transducer.joint import TransducerJoint
from wenet.transducer.predictor import RNNPredictor
from wenet.transformer.cmvn import GlobalCMVN
from wenet.transformer.encoder import ConformerEncoder, TransformerEncoder
from wenet.utils.cmvn import load_cmvn
from wenet.utils.common import IGNORE_ID, add_sos_eos


class Transducer(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        blank_id: int,
        encoder: nn.Module,
        predictor: nn.Module,
        joint: nn.Module,
        ignore_id: int = IGNORE_ID,
    ) -> None:
        assert check_argument_types()
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

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ):
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
                                               reduction="none")
        return loss.sum()

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

    joint = TransducerJoint(vocab_size, **configs['joint_conf'])

    model = Transducer(
        vocab_size=vocab_size,
        blank_id=configs["blank_id"],
        predictor=predictor,
        encoder=encoder,
        joint=joint,
    )
    return model
