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
        self.vocab_size = vocab_size
        self.criterion_att = torch.nn.CrossEntropyLoss(ignore_index=ignore_id, 
                                                       reduction='sum' if length_normalized_loss else 'mean', 
                                                       label_smoothing=lsm_weight)
        self.tie_word_embedding = tie_word_embedding
        self.ignore_id = ignore_id

        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            self.freeze_parameters(self.encoder)
            self.encoder.eval()
        self.freeze_decoder = freeze_decoder
        if freeze_decoder:
            self.freeze_parameters(self.decoder)
            self.decoder.eval()
        self.freeze_llm_embed = freeze_llm_embed
        if freeze_llm_embed:
            self.freeze_parameters(self.embed)
            self.freeze_parameters(self.out)
    
    def train(self, mode: bool = True):
        self.bottleneck.train(mode)
        if not self.freeze_encoder:
            self.encoder.train(mode)
        if not self.freeze_decoder:
            self.encoder.train(mode)
    
    def freeze_parameters(self, moudle: torch.nn.Module):
        for _, param in moudle.named_parameters():
                param.requires_grad = False

    def extract_audio_features(self, audio, audio_lengths):
        output, masks = self.encoder(audio, audio_lengths)
        output, sub_lengths = self.bottleneck(output, masks.sum(-1))
        return output, sub_lengths

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

        # | prefix_embeds | audio_embeds | suffix_embeds | paddings | 
        b, c = prefix_tokens_embeds.size(0), prefix_tokens_embeds.size(2)
        prefix_t = prefix_tokens_embeds.size(1)
        audio_t = audio_embeds.size(1)
        suffix_t = suffix_tokens_embeds.size(1)
        inputs_lengths = prefix_t + audio_t + suffix_t
        input_embeds = torch.ones([b, inputs_lengths, c], device=device)
        targets = torch.ones([b, inputs_lengths], dtype=torch.long, device=device)
        for i in range(b):
            index = 0
            input_embeds[i, :prefix_tokens_lengths[i]] = prefix_tokens_embeds[i, :prefix_tokens_lengths[i]]
            targets[i, :prefix_tokens_lengths[i]-1] = prefix_target[i, :prefix_tokens_lengths[i]-1]

            index += prefix_tokens_lengths[i]
            input_embeds[i, index:index + audio_lengths[i]] = audio_embeds[i, :audio_lengths[i]]
            targets[i, index-1:index + audio_lengths[i]-1] = self.ignore_id

            index += audio_lengths[i]
            input_embeds[i, index:index + suffix_tokens_lengths[i]] = suffix_tokens_embeds[i, :suffix_tokens_lengths[i]]
            targets[i, index-1:index + suffix_tokens_lengths[i]] = suffix_target[i, :suffix_tokens_lengths[i]+1]

            index += suffix_tokens_lengths[i]
            input_embeds[i, index:] = torch.cat([prefix_tokens_embeds[i, prefix_tokens_lengths[i]:],
                                                  audio_embeds[i, audio_lengths[i]:],
                                                    suffix_tokens_embeds[i, suffix_tokens_lengths[i]:]], dim=0)
            targets[i,index:] = self.ignore_id

        mask = ~make_pad_mask(audio_lengths + prefix_tokens_lengths + suffix_tokens_lengths, 
                              max_len=inputs_lengths,
                              pad_type="right").unsqueeze(
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
        batch: dict,
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

        prefix_tokens = batch['prefix_tokens'].to(device)
        audio_feats = batch['audio_feats'].to(device)
        suffix_tokens = batch['suffix_tokens'].to(device)
        prefix_tokens_lengths = batch['prefix_tokens_lengths'].to(device)
        audio_feats_lengths = batch['audio_feats_lengths'].to(device)
        suffix_tokens_lengths = batch['suffix_tokens_lengths'].to(device)

        audio_embeds, audio_lengths = self.extract_audio_features(audio_feats, audio_feats_lengths)

        prefix_tokens_embeds = self.embed(prefix_tokens)
        suffix_tokens_embeds = self.embed(suffix_tokens)

        b, c = prefix_tokens_embeds.size(0), prefix_tokens_embeds.size(2)
        input_embeds_list = []
        token_ids_list = []
        for i in range(b):
            input_embeds = []
            token_ids = []
            input_embeds.append(prefix_tokens_embeds[i, :prefix_tokens_lengths[i]])
            token_ids.append(prefix_tokens[i, :prefix_tokens_lengths[i]])
            input_embeds.append(audio_embeds[i, :audio_lengths[i]])
            token_ids.append(torch.full((1, audio_lengths[i]),
                                      IGNORE_ID,
                                      dtype=torch.int64,
                                      device=device).squeeze(0))
            input_embeds.append(suffix_tokens_embeds[i, :suffix_tokens_lengths[i]])
            token_ids.append(suffix_tokens[i, :suffix_tokens_lengths[i]])
            input_embeds = torch.cat(input_embeds, dim=0)
            token_ids = torch.cat(token_ids, dim=0)
            input_embeds_list.append(input_embeds)
            token_ids_list.append(token_ids)

        min_prompt_len = min(p.shape[0] for p in token_ids_list)
        max_prompt_len = max(p.shape[0] for p in token_ids_list)
        max_seq_len = max_prompt_len + output_len
        assert max_seq_len <= self.decoder.pos_enc.max_len

        # build KV caches
        kv_caches = []
        for _ in range(len(self.decoder.decoders)):
            size = (b, 0, self.decoder.n_kv_head,
                    self.decoder.head_dim)
            k_cache = torch.zeros(size=size, dtype=dtype, device=device)
            v_cache = torch.zeros(size=size, dtype=dtype, device=device)
            kv_caches.append((k_cache, v_cache))

        # prepare inputs
        token_ids_tensor = torch.full((b, max_seq_len),
                                      IGNORE_ID,
                                      dtype=torch.int64,
                                      device=device)
        input_embeds_tensor = torch.zeros((b, min_prompt_len, c),
                                            dtype=dtype,
                                            device=device)
        # right padding
        for i, (embeds, tokens) in enumerate(zip(input_embeds_list, token_ids_list)):
            token_ids_tensor[i, :len(tokens)] = tokens
            input_embeds_tensor[i, :min_prompt_len] = embeds[:min_prompt_len]

        prompt_mask_tensor = ~make_pad_mask(audio_lengths + prefix_tokens_lengths + suffix_tokens_lengths,
                              max_len=max_seq_len)
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
            [temperature] * b).to(device)
        top_ps_tensor = torch.FloatTensor([top_p] * b).to(device)
        top_ks_tensor = torch.LongTensor([top_k] * b).to(device)
        output_index = torch.tensor(min_prompt_len,
                                    dtype=torch.int64).to(device)

        offset = torch.tensor([0] * b).to(device)
        input_offset = offset

        stop_tokens_tensor = torch.tensor(stop_tokens, device=device)
        # Prefill up to min_prompt_len tokens, then treat other prefill as
        # decode and ignore output.
        for i in range(max_seq_len - min_prompt_len):
            decoder_out, kv_caches, = self.decoder(
                input_embeds_tensor,
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
            input_embeds_tensor = self.embed(input_token_ids_tensor)

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
            trimmed_output = tokens[len(token_ids_list[i]
                                        ):len(token_ids_list[i]) + output_len]
            for stop_token in stop_tokens:
                try:
                    eos_index = trimmed_output.index(stop_token)
                    trimmed_output = trimmed_output[:eos_index]
                    break
                except Exception:
                    continue
            results.append(trimmed_output)

        return results