import torch
from wenet.utils.common import log_add
from typing import Tuple

class Sequence():

    __slots__ = {'hyp', 'score', 'h_0', 'c_0'}

    def __init__(
        self,
        hyp: torch.Tensor,
        score,
        h_0: torch.Tensor,
        c_0: torch.Tensor,
        last,
    ):
        self.hyp = hyp
        self.score = score
        self.h_0 = h_0
        self.c_0 = c_0

class PrefixBeamSearch():
    def __init__(
        self,
        encoder,
        predictor,
        joint,
        ctc,
        sos,
        blank
    ):
        self.encoder = encoder
        self.predictor = predictor
        self.joint = joint
        self.ctc = ctc
        self.sos = sos
        self.blank = blank

    def forward_decoder_one_step(
        self,
        encoder_x: torch.Tensor,
        pre_t: torch.Tensor,
        h_0: torch.Tensor,
        c_0: torch.Tensor,
    ):
        pre_t, h_1, c_1 = self.predictor.forward_step(
            pre_t.unsqueeze(-1),
            None,
            h_0,
            c_0
        )
        x = self.joint(encoder_x, pre_t)
        x = x.log_softmax(dim=-1)
        return x, (h_1, c_1)

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

    def prefix_beam_search(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        decoding_chunk_size: int = -1,
        beam_size: int = 5,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
        ctc_weight: float = 0.3,
        transducer_weight: float = 0.7
    ):
        """prefix beam search
           also see wenet.transducer.transducer.beam_search
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        device = speech.device
        batch_size = speech.shape[0]
        assert batch_size == 1

        # 1. Encoder
        encoder_out, encoder_mask = self.encoder(
            speech, speech_lengths, decoding_chunk_size,
            num_decoding_left_chunks)  # (B, maxlen, encoder_dim)
        maxlen = encoder_out.size(1)
        encoder_dim = encoder_out.size(2)
        running_size = batch_size * beam_size
        predictor_num = self.predictor.n_layers

        ctc_probs = self.ctc.log_softmax(
            encoder_out).squeeze(0)
        beam_init = []

        # 2. init beam using Sequence to save beam unit
        h_0, c_0 = self.predictor.init_state(1, method="zero")
        beam_init.append(
            Sequence(
                hyp=[self.blank],
                score=0.0,
                h_0=h_0,
                c_0=c_0,
            )
        )
        # 3. start decoding (notice: we use breathwise first searching)
        # !!!! In this decoding method: one frame do not output multi units. !!!!
        # !!!!    Experiments show that this strategy has little impact      !!!!
        for i in range(maxlen):
            # 3.1 building input
            # decoder taking the last token to predict the next token
            input_hyp = [s.hyp[-1] for s in beam_init]
            input_hyp_tensor = torch.tensor(input_hyp, dtype=torch.int, device=device)
            # building statement from beam
            h_0 = torch.concat(
                [s.h_0 for s in beam_init], dim=1
            ).to(device)
            c_0 = torch.concat(
                [s.c_0 for s in beam_init], dim=1
            ).to(device)
            # build score tensor to do torch.add() function
            scores = torch.tensor(
                [s.score for s in beam_init]
            ).to(device)

            # 3.2 forward decoder
            logp, (h_1, c_1) = self.forward_decoder_one_step(
                encoder_out[:, i, :].unsqueeze(1), input_hyp_tensor, h_0, c_0
            )  # logp: (N, 1, 1, vocab_size)
            logp = logp.squeeze(1).squeeze(1)  # logp: (N, vocab_size)

            # 3.3 shallow fusion for transducer score
            #     and ctc score where we can also add the LM score
            logp = torch.log(
                torch.add(
                    transducer_weight * torch.exp(logp),
                    ctc_weight * torch.exp(ctc_probs[i].unsqueeze(0))
                )
            )

            # 3.4 first beam prune
            top_k_logp, top_k_index = logp.topk(beam_size)  # (N, N)
            scores = torch.add(scores.unsqueeze(1), top_k_logp)

            # 3.5 generate new beam (N*N)
            beam_A = []
            for j in range(len(beam_init)):
                # update seq
                base_seq = beam_init[j]
                for t in range(beam_size):
                    # blank: only update the score
                    if top_k_index[j, t] == self.blank:
                        new_seq = Sequence(
                            hyp=base_seq.hyp.copy(),
                            score=scores[j, t].item(),
                            h_0=h_0[:, j, :].unsqueeze(1),
                            c_0=c_0[:, j, :].unsqueeze(1),
                        )

                        beam_A.append(new_seq)


                    # other unit: update hyp score statement
                    else:
                        hyp_new = base_seq.hyp.copy()
                        hyp_new.append(top_k_index[j, t].item())
                        new_seq = Sequence(
                            hyp=hyp_new,
                            score=scores[j, t].item(),
                            h_0=h_1[:, j, :].unsqueeze(1),
                            c_0=c_1[:, j, :].unsqueeze(1),
                        )
                        beam_A.append(new_seq)

            # 3.6 prefix fusion
            fusion_A = [beam_A[0]]
            for j in range(1, len(beam_A)):
                s1 = beam_A[j]
                if_do_append = True
                for t in range(len(fusion_A)):
                    # notice: A_ can not fusion with A
                    if s1.hyp == fusion_A[t].hyp :
                        fusion_A[t].score = log_add([fusion_A[t].score, s1.score])
                        if_do_append = False
                        break
                if if_do_append:
                    fusion_A.append(s1)

            # 4. second pruned
            fusion_A.sort(key=lambda x: x.score, reverse=True)
            beam_init = fusion_A[:beam_size]
        return beam_init, encoder_out
