from typing import Dict, List, Optional, Union
import torch
from wenet.LLM.sampler import sampler
from wenet.transformer.encoder import TransformerEncoder
from wenet.AudioLLM.bottleneck import ConvLinearBottleNeck
from wenet.LLM.decoder import DecoderOnly
from wenet.utils.common import IGNORE_ID, th_accuracy
from wenet.utils.mask import make_pad_mask, subsequent_mask


class AudioLLM(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        encoder: TransformerEncoder,
        decoder: DecoderOnly,
        special_tokens: dict,
        tie_word_embedding: bool = False,
        linear_bias: bool = False,
        ignore_id: int = IGNORE_ID,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        bottleneck_type: str = "conv-linear",
        freeze_encoder: bool = True,
        freeze_llm_embed: bool = True,
        freeze_decoder: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        del special_tokens
        self.encoder = encoder
        self.decoder = decoder
        self.embed = torch.nn.Embedding(vocab_size, decoder.hidden_size)
        self.out = torch.nn.Linear(decoder.hidden_size,
                                   vocab_size,
                                   bias=linear_bias)
        if bottleneck_type == "conv-linear":
            self.bottleneck = ConvLinearBottleNeck(encoder.output_size(), decoder.hidden_size, **kwargs)
        self.speech_ln = torch.nn.LayerNorm(decoder.hidden_size)
        self.vocab_size = vocab_size
        self.criterion_att = torch.nn.CrossEntropyLoss(ignore_index=ignore_id, 
                                                       reduction='sum' if length_normalized_loss else 'mean', 
                                                       label_smoothing=lsm_weight)
        self.tie_word_embedding = tie_word_embedding
        self.ignore_id = ignore_id

        if freeze_encoder:
            self.freeze_parameters(self.encoder)
            self.encoder.eval()

        if freeze_decoder:
            self.freeze_parameters(self.decoder)
            self.decoder.eval()

        if freeze_llm_embed:
            self.freeze_parameters(self.embed)
            self.freeze_parameters(self.out)

    def extract_audio_features(self, audio, audio_lengths):
        output, masks = self.encoder(audio, audio_lengths)
        output, sub_lengths = self.bottleneck(output, masks.sum(-1))
        return self.speech_ln(output), sub_lengths
    
    def freeze_parameters(self, moudle: torch.nn.Module):
        for _, param in moudle.named_parameters():
                param.requires_grad = False

    @torch.jit.unused
    def forward(
        self,
        batch: dict,
        device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """ Forward for training
        """
        prefix_tokens = batch['prefix_tokens'].to(device)
        audio_feats = batch['audio_feats'].to(device)
        suffix_tokens = batch['suffix_tokens'].to(device)
        prefix_target = batch['prefix_target'].to(device)
        suffix_target = batch['suffix_target'].to(device)
        prefix_tokens_lengths = batch['prefix_tokens_lengths'].to(device)
        audio_feats_lengths = batch['audio_feats_lengths'].to(device)
        suffix_tokens_lengths = batch['suffix_tokens_lengths'].to(device)

        audio_embeds, audio_lengths = self.extract_audio_features(audio_feats, audio_feats_lengths)

        prefix_tokens_embeds = self.embed(prefix_tokens)
        suffix_tokens_embeds = self.embed(suffix_tokens)

        # token padding | audio padding | prefix_embeds | audio_embeds | suffix_embeds
        #              \_prefix_padding | suffix_padding
        b, c = prefix_tokens_embeds.size(0), prefix_tokens_embeds.size(2)
        prefix_t = prefix_tokens_embeds.size(1)
        audio_t = audio_embeds.size(1)
        suffix_t = suffix_tokens_embeds.size(1)
        inputs_lengths = prefix_t + audio_t + suffix_t
        input_embeds = torch.ones([b, inputs_lengths, c], device=device)
        targets = torch.ones([b, inputs_lengths], dtype=torch.long, device=device)

        for i in range(b):
            index = 0
            input_embeds[i,index:prefix_t - prefix_tokens_lengths[i]] = prefix_tokens_embeds[i, prefix_tokens_lengths[i]:]
            index += prefix_t - prefix_tokens_lengths[i]
            input_embeds[i, index:index + audio_t - audio_lengths[i]] = audio_embeds[i, audio_lengths[i]:]
            index += audio_t - audio_lengths[i]
            input_embeds[i, index:index + suffix_t - suffix_tokens_lengths[i]] = suffix_tokens_embeds[i, suffix_tokens_lengths[i]:]
            index += suffix_t - suffix_tokens_lengths[i]
            targets[i,:index] = self.ignore_id
            input_embeds[i, index:index + prefix_tokens_lengths[i]] = prefix_tokens_embeds[i, :prefix_tokens_lengths[i]]
            targets[i, index:index + prefix_tokens_lengths[i]-1] = prefix_target[i, :prefix_tokens_lengths[i]]
            index += prefix_tokens_lengths[i]
            input_embeds[i, index:index + audio_lengths[i]] = audio_embeds[i, :audio_lengths[i]]
            targets[i, index -1:index -1 + audio_lengths[i]] = self.ignore_id
            index += audio_lengths[i]
            input_embeds[i, index:index + suffix_tokens_lengths[i]] = suffix_tokens_embeds[i, :suffix_tokens_lengths[i]]
            targets[i, index-1:index + suffix_tokens_lengths[i]] = suffix_target[i, :suffix_tokens_lengths[i] + 1]
        mask = ~make_pad_mask(audio_lengths + prefix_tokens_lengths + suffix_tokens_lengths, 
                              max_len=inputs_lengths,
                              pad_type="left").unsqueeze(
            1)  # (B,1,L)

        causal_mask = subsequent_mask(
            mask.size(-1), device=mask.device).unsqueeze(0)  # (1,L,L)
        att_mask = causal_mask & mask  # (B, L, L)

        decoder_out = self.out(self.decoder(input_embeds,
                                            att_mask)[0]) # (B, L, vocab_size)
        loss = self.criterion_att(decoder_out.view(-1, self.vocab_size), 
                                  targets.view(-1))
        acc = th_accuracy(decoder_out.view(-1, self.vocab_size),
                          targets,
                          ignore_label=self.ignore_id)

        return {
            "loss": loss,
            "ppl": torch.exp(loss.detach()),
            "th_accuracy": acc
        }

    def tie_or_clone_weights(self, jit_mode: bool):
        if not self.tie_word_embedding:
            return
        if jit_mode:
            self.out.weight = torch.nn.Parameter(self.embed.weight.clone())
        else:
            self.out.weight = self.embed.weight
            # TODO(Mddct): whether to deal bias for other llm model

    @torch.jit.unused
    @torch.inference_mode()
    def generate(
        self,
        prompts_tokens: List[List[int]],
        device: torch.device,
        stop_tokens: List[int],
        dtype: torch.dtype = torch.float32,
        output_len: int = 100,
        temperature: Union[float, None] = 0.95,
        top_p: float = 1.0,
        top_k: int = 100,
    ) -> List[List[int]]:
        """Generates responses for given prompts using Gemma model."""
        # If a single prompt is provided, treat it as a batch of 1.
        batch_size = len(prompts_tokens)
        min_prompt_len = min(len(p) for p in prompts_tokens)
        max_prompt_len = max(len(p) for p in prompts_tokens)
        max_seq_len = max_prompt_len + output_len
        assert max_seq_len <= self.decoder.pos_enc.max_len

        # build KV caches
        kv_caches = []
        for _ in range(len(self.decoder.decoders)):
            size = (batch_size, 0, self.decoder.n_kv_head,
                    self.decoder.head_dim)
            k_cache = torch.zeros(size=size, dtype=dtype, device=device)
            v_cache = torch.zeros(size=size, dtype=dtype, device=device)
            kv_caches.append((k_cache, v_cache))

        # prepare inputs
        token_ids_tensor = torch.full((batch_size, max_seq_len),
                                      IGNORE_ID,
                                      dtype=torch.int64,
                                      device=device)
        input_token_ids_tensor = torch.full((batch_size, min_prompt_len),
                                            IGNORE_ID,
                                            dtype=torch.int64,
                                            device=device)
        # right padding
        for i, p in enumerate(prompts_tokens):
            token_ids_tensor[i, :len(p)] = torch.tensor(p)
            input_token_ids_tensor[i, :min_prompt_len] = torch.tensor(
                p[:min_prompt_len])

        prompt_mask_tensor = token_ids_tensor != IGNORE_ID
        input_positions_tensor = torch.arange(0,
                                              min_prompt_len,
                                              dtype=torch.int64).to(device)
        mask_tensor = torch.ones((1, 1, max_seq_len, max_seq_len),
                                 dtype=torch.bool)
        mask_tensor = torch.tril(mask_tensor).to(device)
        curr_mask_tensor = mask_tensor.index_select(2, input_positions_tensor)
        att_mask = curr_mask_tensor.squeeze(
            1)[:, :min_prompt_len, :min_prompt_len]
        output_positions_tensor = torch.LongTensor([min_prompt_len - 1
                                                    ]).to(device)
        temperatures_tensor = None if not temperature else torch.FloatTensor(
            [temperature] * batch_size).to(device)
        top_ps_tensor = torch.FloatTensor([top_p] * batch_size).to(device)
        top_ks_tensor = torch.LongTensor([top_k] * batch_size).to(device)
        output_index = torch.tensor(min_prompt_len,
                                    dtype=torch.int64).to(device)

        input_token_embeding = self.embed(input_token_ids_tensor)
        offset = torch.tensor([0] * len(prompts_tokens)).to(device)
        input_offset = offset

        stop_tokens_tensor = torch.tensor(stop_tokens, device=device)
        # Prefill up to min_prompt_len tokens, then treat other prefill as
        # decode and ignore output.
        for i in range(max_seq_len - min_prompt_len):
            decoder_out, kv_caches, = self.decoder(
                input_token_embeding,
                att_mask,
                input_offset,
                kv_caches,
            )
            decoder_out = self.out(decoder_out)
            decoder_out = decoder_out.index_select(1, output_positions_tensor)
            next_token_ids = sampler(
                decoder_out,
                temperatures_tensor,
                top_ps_tensor,
                top_ks_tensor,
            )
            curr_prompt_mask = prompt_mask_tensor.index_select(
                1, output_index).squeeze(dim=1)
            curr_token_ids = token_ids_tensor.index_select(
                1, output_index).squeeze(dim=1)
            output_token_ids = torch.where(curr_prompt_mask, curr_token_ids,
                                           next_token_ids).unsqueeze(dim=1)
            token_ids_tensor.index_copy_(1, output_index, output_token_ids)

            input_token_ids_tensor = output_token_ids
            input_token_embeding = self.embed(input_token_ids_tensor)

            input_positions_tensor = output_index.unsqueeze(dim=-1)
            curr_mask_tensor = mask_tensor.index_select(
                2, input_positions_tensor)
            att_mask = curr_mask_tensor.squeeze(1)[:, :output_index +
                                                   1, :output_index + 1]

            output_positions_tensor = torch.tensor(
                0, dtype=torch.int64).to(device)
            input_offset = offset + output_index.unsqueeze(-1)
            output_index = output_index + 1

            if all(torch.isin(next_token_ids, stop_tokens_tensor)):
                break

        token_ids = token_ids_tensor.tolist()
        results = []
        for i, tokens in enumerate(token_ids):
            trimmed_output = tokens[len(prompts_tokens[i]
                                        ):len(prompts_tokens[i]) + output_len]
            for stop_token in stop_tokens:
                try:
                    eos_index = trimmed_output.index(stop_token)
                    trimmed_output = trimmed_output[:eos_index]
                    break
                except Exception:
                    continue
            results.append(trimmed_output)

        return results