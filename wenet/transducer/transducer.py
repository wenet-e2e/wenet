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
    vocab_size = configs['output_dim']

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
