# Copyright 2022 FBK
import math

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import torch
from torch import Tensor
from typing import List, Dict, Optional

from examples.speech_to_text.models.dual_encoder import FairseqDualEncoderModel
from fairseq.sequence_generator import SequenceGenerator, EnsembleModel


class DualEncoderSequenceGenerator(SequenceGenerator):
    def __init__(
        self,
        models,
        tgt_dict,
        beam_size=1,
        max_len_a=0,
        max_len_b=200,
        min_len=1,
        normalize_scores=True,
        len_penalty=1.0,
        unk_penalty=0.0,
        temperature=1.0,
        match_source_len=False,
        no_repeat_ngram_size=0,
        search_strategy=None,
        eos=None,
        symbols_to_strip_from_output=None,
        lm_model=None,
        lm_weight=1.0,
    ):
        """Generates translations of a given source sentence.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models,
                currently support fairseq.models.TransformerModel for scripting
            beam_size (int, optional): beam width (default: 1)
            max_len_a/b (int, optional): generate sequences of maximum length
                ax + b, where x is the source length
            min_len (int, optional): the minimum length of the generated output
                (not including end-of-sentence)
            normalize_scores (bool, optional): normalize scores by the length
                of the output (default: True)
            len_penalty (float, optional): length penalty, where <1.0 favors
                shorter, >1.0 favors longer sentences (default: 1.0)
            unk_penalty (float, optional): unknown word penalty, where <0
                produces more unks, >0 produces fewer (default: 0.0)
            temperature (float, optional): temperature, where values
                >1.0 produce more uniform samples and values <1.0 produce
                sharper samples (default: 1.0)
            match_source_len (bool, optional): outputs should match the source
                length (default: False)
        """
        super().__init__(models, tgt_dict, beam_size, max_len_a, max_len_b, min_len,
                         normalize_scores, len_penalty, unk_penalty, temperature, match_source_len,
                         no_repeat_ngram_size, search_strategy, eos, symbols_to_strip_from_output,
                         lm_model, lm_weight)
        if isinstance(models, DualEncoderEnsembleModel):
            self.model = models
        else:
            self.model = DualEncoderEnsembleModel(models)
        self.model.eval()


class DualEncoderEnsembleModel(EnsembleModel):
    def __init__(self, models):
        for m in models:
            assert isinstance(m, FairseqDualEncoderModel)
        super().__init__(models)

    @torch.jit.export
    def forward_encoder(self, net_input):
        if not self.has_encoder():
            return None
        encoder_outs = [model.encoder.forward_torchscript(net_input) for model in self.models]
        context_outs = [
            model.context_encoder.forward(net_input["context_tokens"], src_lengths=net_input["context_lengths"])
            for model in self.models
        ]

        return [[enc_out, context_out] for enc_out, context_out in zip(encoder_outs, context_outs)]

    @torch.jit.export
    def forward_decoder(
        self,
        tokens,
        encoder_outs: List[List[Dict[str, List[Tensor]]]],
        incremental_states: List[Dict[str, Dict[str, Optional[Tensor]]]],
        temperature: float = 1.0,
    ):
        log_probs = []
        avg_attn: Optional[Tensor] = None
        encoder_out: Optional[Dict[str, List[Tensor]]] = None
        context_out: Optional[Dict[str, List[Tensor]]] = None
        for i, model in enumerate(self.models):
            if self.has_encoder():
                encoder_out = encoder_outs[i][0]
                context_out = encoder_outs[i][1]
            # decode each model considering both main and context encoder
            if self.has_incremental_states():
                decoder_out = model.decoder.forward(
                    tokens,
                    encoder_out=encoder_out,
                    context_encoder_out=context_out,
                    incremental_state=incremental_states[i],
                )
            else:
                decoder_out = model.decoder.forward(tokens, encoder_out=encoder_out, context_encoder_out=context_out)

            attn: Optional[Tensor] = None
            decoder_len = len(decoder_out)
            if decoder_len > 1 and decoder_out[1] is not None:
                if isinstance(decoder_out[1], Tensor):
                    attn = decoder_out[1]
                else:
                    attn_holder = decoder_out[1]["attn"]
                    if isinstance(attn_holder, Tensor):
                        attn = attn_holder
                    elif attn_holder is not None:
                        attn = attn_holder[0]
                if attn is not None:
                    attn = attn[:, -1, :]

            decoder_out_tuple = (
                decoder_out[0][:, -1:, :] if temperature == 1. else decoder_out[0][:, -1:, :].div_(temperature),
                None if decoder_len <= 1 else decoder_out[1],
            )
            probs = model.get_normalized_probs(
                decoder_out_tuple, log_probs=True, sample=None
            )
            probs = probs[:, -1, :]
            if self.models_size == 1:
                return probs, attn

            log_probs.append(probs)
            if attn is not None:
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)

        avg_probs = torch.logsumexp(torch.stack(log_probs, dim=0), dim=0) - math.log(
            self.models_size
        )

        if avg_attn is not None:
            avg_attn.div_(self.models_size)
        return avg_probs, avg_attn

    def reorder_encoder_out(self, encoder_outs, new_order):
        """
        Reorder encoder output of both main and context encoders according to *new_order*.

        Args:
            encoder_outs: output from the ``forward()`` method of both main and context encoders
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        new_outs = []
        if not self.has_encoder():
            return new_outs
        for i, model in enumerate(self.models):
            assert encoder_outs is not None
            new_outs.append([
                model.encoder.reorder_encoder_out(encoder_outs[i][0], new_order),
                model.context_encoder.reorder_encoder_out(encoder_outs[i][1], new_order)
            ])
        return new_outs
