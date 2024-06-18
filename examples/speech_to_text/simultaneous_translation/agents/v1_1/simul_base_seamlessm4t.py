# Copyright 2024 FBK

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

from typing import Optional
from simuleval.agents.states import AgentStates
from simuleval.utils import entrypoint
from simuleval.agents import SpeechToTextAgent
from simuleval.agents.actions import WriteAction

from examples.speech_to_text.simultaneous_translation.agents.speech_utils import BOW_PREFIX
from fairseq.utils import import_user_module

import numpy as np
import torch


@entrypoint
class SimulSeamlessS2T(SpeechToTextAgent):
    """
    Base class for Simultaneous agents based on the
    `SeamlessM4T <https://ai.meta.com/blog/seamless-m4t/>`_ architecture.
    The huggingface implementation is used.
    """
    def __init__(self, args):
        SpeechToTextAgent.__init__(self, args)
        self.source_segment_size = args.source_segment_size
        model_name = "facebook/seamless-m4t-" + args.model_size
        self.device = args.device
        fp16 = args.fp16
        self.load_model(model_name, fp16)
        self.num_beams = args.num_beams
        self.max_new_tokens = args.max_new_tokens
        self.max_len = args.max_len
        self.tgt_lang = args.target_language
        if len(self.tgt_lang) != 3:
            raise ValueError("`target_language` must be of 3 characters.")

    def load_model(self, model_name, fp16):
        """
        Loads the SeamlessM4T model in Hugging Face format.
        """
        from transformers import AutoProcessor, SeamlessM4Tv2Model, AutoTokenizer, SeamlessM4TModel
        precision = torch.float16 if fp16 else torch.float32
        if "v2" in model_name:
            self.model = SeamlessM4Tv2Model.from_pretrained(
                model_name, device_map=self.device, torch_dtype=precision)
        else:
            model_name = model_name.replace("seamless", "hf-seamless")
            self.model = SeamlessM4TModel.from_pretrained(
                model_name, device_map=self.device, torch_dtype=precision)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.sampling_rate = self.processor.feature_extractor.sampling_rate

    @staticmethod
    def add_args(parser):
        parser.add_argument("--target-language", default="eng", type=str)
        parser.add_argument("--model-size", default="v2-large", type=str)
        parser.add_argument(
            "--max-len", type=int, default=200, help="Max length of textual output")
        parser.add_argument(
            "--num-beams", default=5, type=int, help="Number of beams of the beam search")
        parser.add_argument(
            "--max-new-tokens", type=int, default=128, help="The maximum numbers of tokens to generate")

    def policy(self, states: Optional[AgentStates] = None):
        if states is None:
            states = self.states

        # If the input has been completely received, write the remaining hypothesis
        if states.source_finished:
            action = self._emit_remaining_tokens(states)
        else:
            action = self._policy(states)
        return action

    def _policy(self, states: Optional[AgentStates] = None):
        """
        Determines whether to read more input (returning a `ReadAction`) or emit a full or partial translation
        (returning a `WriteAction`). It represents the logic of the policy.
        """
        pass

    def get_input_features(self, states):
        """
        Extracts the input features requested by SeamlessM4T from the waveform stored in the
        `source` property of `states`.
        It encodes the whole audio each time a new audio chunk is received and returns the
        encoded features.
        """
        torch.cuda.empty_cache()
        input_features = (self.processor(
            audios=np.array(states.source).astype("float32"),
            return_tensors="pt",
            sampling_rate=self.sampling_rate).to(self.device))["input_features"]
        return input_features

    def get_prefix(self, states):
        """
        Creates a prefix for the generation phase with the previous outputs. The prefix is formatted
        following the SeamlessM4T template (tgt language token idx + prefix).
        """
        prefix_ids = torch.tensor(
            [[self.model.generation_config.text_decoder_lang_to_code_id.get(self.tgt_lang)]]
        ).to(self.device)

        if len(states.target) > 0:
            prev_out = " ".join(states.target).strip()
            prev_out_ids = self.tokenizer(
                prev_out, add_special_tokens=False, padding=False,
                return_tensors="pt")["input_ids"].to(self.device)
            prefix_ids = torch.cat((prefix_ids, prev_out_ids), dim=1).to(self.device)
        return prefix_ids.long()

    def _generate(self, input_features, prefix_ids):
        """
        Generates a new hypothesis forcing the `prefix_ids` prefix tokens.
        """
        gen_out = self.model.generate(
            input_features=input_features,
            decoder_input_ids=prefix_ids,
            num_beams=self.num_beams,
            max_new_tokens=self.max_new_tokens,
            no_repeat_ngram_size=5,
            return_dict_in_generate=True,
            generate_speech=False)
        out_tokens = gen_out[0].tolist()[0]
        return out_tokens

    def generate_and_decode(self, input_features, prefix_ids) -> str:
        """
        Generates a new hypothesis returning is as a string.
        """
        out_tokens = self._generate(input_features, prefix_ids)
        out_tokens = out_tokens[prefix_ids.shape[1] + 1:]

        # Get textual prediction without special tokens
        prediction = self.tokenizer.decode(out_tokens, skip_special_tokens=True)
        return prediction

    def generate_with_cross_attn(self, input_features, prefix_ids, layer, normalize_attn=True):
        """
        Generates a new hypothesis returning also the cross attention scores.
        The hypothesis is forced to start with the `prefix_ids` prefix tokens, which are then
        removed from both the returned output tokens and the attention scores.  
        """
        gen_out = self.model.generate(
            input_features=input_features,
            decoder_input_ids=prefix_ids,
            num_beams=self.num_beams,
            output_attentions=True,
            max_new_tokens=self.max_new_tokens,
            no_repeat_ngram_size=5,
            return_dict_in_generate=True,
            generate_speech=False)
        out_tokens = list(gen_out.sequences[0])

        # Exclude BOS, prefix, and EOS from the generated sequence
        new_hypo = out_tokens[prefix_ids.shape[1] + 1:-1]
        cross_attn_scores = None
        if len(new_hypo) > 0:
            cross_attn_scores = self.get_cross_attention(
                gen_out, len(new_hypo), layer, normalize_attn=True)

        return new_hypo, cross_attn_scores

    def get_cross_attention(self, gen_out, new_hypo_len, layer, normalize_attn):
        """
        Given the attention matrices for each generation step and layer, the beam indices selected
         at each step of the beam search (if num_beams > 1), and the length of the new hypotheses,
         this function returns the cross attention scores from Layer *layer* by averaging the
         scores along the attention heads dimension.
         The first element of *cross_attentions* is skipped since it contains the prefix while we
         are interested in the cross attention matrices of the new predicted hypotheses.
        """
        cross_attns = []
        if self.num_beams > 1:
            # Beam search: for each token of the new hypothesis, we select the corresponding cross
            # attention from the cross attentions stored at each step of the beam search using the
            # index contained in the tensor of indices beam_indices (num_beams * sequence length)
            for tok_idx in range(new_hypo_len):
                # Select the cross attention matrix using the beam_indices
                beam_indices = gen_out.beam_indices[:, tok_idx]
                # add some comments on why tok_idx + 1, and the -1 selection
                cross_attn = gen_out.cross_attentions[tok_idx + 1][layer][:, :, -1, :]
                cross_attn = cross_attn.index_select(dim=0, index=beam_indices)
                cross_attns.append(cross_attn)
        else:
            # Greedy search
            for tok_idx in range(new_hypo_len):
                cross_attn = gen_out.cross_attentions[tok_idx + 1][layer][:, :, -1, :]
                cross_attns.append(cross_attn)

        # Cross attention scores with shape [num_heads, sequence_len, n_audio_features]
        cross_attns = torch.stack(cross_attns)

        # Average on the attention heads dimension
        cross_attns = cross_attns.squeeze(1).mean(dim=1)

        # Normalize attention scores
        if normalize_attn:
            cross_attns = self.normalize_attn(cross_attns)
        return cross_attns

    @staticmethod
    def normalize_attn(attn):
        std = attn.std(axis=0)
        std[std == 0.] = 1.0
        mean = attn.mean(axis=0)
        return (attn - mean) / std

    @staticmethod
    def get_words(tokens):
        # Remove last incomplete word(s)
        num_tokens_incomplete = 0
        for tok in reversed(tokens):
            num_tokens_incomplete += 1
            if tok.startswith(BOW_PREFIX):
                tokens_to_write = " ".join(tokens[:-num_tokens_incomplete])
                return tokens_to_write.replace(" ", "").replace(BOW_PREFIX, " ").strip()
        return None

    def _emit_remaining_tokens(self, states):
        """
        It implements the actions to be performed when the input has been completely received.
        The last hypothesis is fully emitted.
        """
        input_features = self.processor(
            audios=np.array(states.source).astype("float32"),
            return_tensors="pt",
            sampling_rate=self.sampling_rate).to(self.device)["input_features"]
        prefix_ids = self.get_prefix(states)
        prediction = self.generate_and_decode(input_features, prefix_ids)

        return WriteAction(
            content=prediction,
            finished=states.source_finished)
