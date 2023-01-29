# Copyright 2023 FBK

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import os
import torch

from examples.speech_to_text.simultaneous_translation.agents.simul_offline_waitk import FairseqSimulSTAgent, \
    TensorListEntry, BOW_PREFIX
from fairseq import checkpoint_utils, tasks
from fairseq.utils import import_user_module
from fairseq.data.audio.speech_to_text_dataset import SpeechToTextDataset

try:
    from simuleval import READ_ACTION, WRITE_ACTION, DEFAULT_EOS
    from simuleval.agents import SpeechAgent
    from simuleval.states import ListEntry, SpeechStates
except ImportError:
    print("Please install simuleval 'pip install simuleval'")
    raise ImportError


class LocalAgreementSimulSTAgent(FairseqSimulSTAgent):
    """
    Local Agreement agent for Simultaneous Speech Translation based on
    "Low-Latency Sequence-to-Sequence Speech Recognition and Translation by Partial Hypothesis Selection"
    (https://www.isca-speech.org/archive/pdfs/interspeech_2020/liu20s_interspeech.pdf)
    by Liu et al., 2020. The agent displays the agreeing prefixes of the two consecutive chunks:
    during the first n−1 chunks, no output is produced; from the n-th chunk on, the longest common prefix
    of the n consecutive chunks is identified and emitted.
    Empirically, the authors found that n=2 works better.
    The implementation works only for SentencePiece up to now.
    """
    def __init__(self, args):
        super().__init__(args)
        # Local Agreement using last 2 generated sentences as memory
        self.la_n = 2
        torch.set_grad_enabled(False)

    @staticmethod
    def add_args(parser):
        # fmt: off
        FairseqSimulSTAgent.add_args(parser)
        parser.add_argument("--speech-segment-factor", type=int, default=1,
                            help="Factor to be multiplied by the speech segment size to obtain the final speech segment"
                                 "size.")
        parser.add_argument("--beam", type=int, default=5,
                            help="Beam size.")
        parser.add_argument("--no-repeat-ngram-size", type=int, default=0,
                            help="Same parameter of the the fairseq-generate.")
        parser.add_argument("--prefix-size", default=0, type=int,
                            help="Initialize generation by target prefix of given length.")
        parser.add_argument("--prefix-token", default="", type=str,
                            help="Target prefix token.")
        # fmt: on
        return parser

    def load_model_vocab(self, args):
        filename = args.model_path
        if not os.path.exists(filename):
            raise IOError("Model file not found: {}".format(filename))

        state = checkpoint_utils.load_checkpoint_to_cpu(filename)

        # task is taken from the checkpoint since SimulEval does not take the task as an argument
        task_args = state["cfg"]["task"]
        task_args.data = args.data_bin

        # if a configuration file (yaml) is passed as argument, it is loaded with the command line args
        if args.config is not None:
            task_args.config_yaml = args.config

        import_user_module(state["cfg"].common)

        self.task = tasks.setup_task(task_args)

        # build model for ensemble
        state["cfg"]["model"].load_pretrained_encoder_from = None
        state["cfg"]["model"].load_pretrained_decoder_from = None
        self.model = self.task.build_model(state["cfg"]["model"])
        self.model.load_state_dict(state["model"], strict=True)
        self.model.eval()
        # share model tensors to let SimulEval processes to interact and exchange information and data
        self.model.share_memory()

        state["cfg"]["generation"]["beam"] = args.beam
        state["cfg"]["generation"]["no_repeat_ngram_size"] = args.no_repeat_ngram_size
        state["cfg"]["generation"]["prefix_size"] = args.prefix_size
        self.generator = self.task.build_generator(
            [self.model], state["cfg"]["generation"]
        )

        if self.gpu:
            self.model.cuda()

        # Set dictionary
        self.tgtdict = self.task.target_dictionary

        self.prefix_token_idx = None
        if args.prefix_token != "":
            lang_tag = SpeechToTextDataset.LANG_TAG_TEMPLATE.format(args.prefix_token)
            self.prefix_token_idx = self.tgtdict.index(lang_tag)

        self.eos_idx = self.tgtdict.index(DEFAULT_EOS)

    def initialize_states(self, states):
        self.feature_extractor.clear_cache()
        states.units.source = TensorListEntry()
        states.units.target = ListEntry()
        states.incremental_states = dict()
        states.chunks_hyp = []
        states.displayed = []
        states.retrieved = []
        states.new_segment = False
        states.write = []
        self.bos = True

    @staticmethod
    def is_there_punct(t):
        return '.' in t or '?' in t or '!' in t

    def units_to_segment(self, units, states):
        # Merge sub word to full word
        if self.eos_idx == units[0]:
            return DEFAULT_EOS

        segment = []
        if None in units.value:
            units.value.remove(None)

        for index in units:
            if index is None:
                units.pop()
            token = self.model.decoder.dictionary[index]
            if token.startswith(BOW_PREFIX) or index == self.eos_idx:
                if len(segment) == 0:
                    if token != DEFAULT_EOS:
                        segment.append(token.replace(BOW_PREFIX, ""))
                    else:
                        segment.append(DEFAULT_EOS)
                else:
                    for j in range(len(segment)):
                        units.pop()

                    string_to_return = ["".join(segment)]
                    if self.bos:
                        string_to_return[0] = string_to_return[0].capitalize()
                        self.bos = False

                    if self.eos_idx == units[0]:
                        string_to_return.append(DEFAULT_EOS)

                    self.bos = self.is_there_punct(string_to_return[-1])

                    return string_to_return
            else:
                segment.append(token.replace(BOW_PREFIX, ""))

        if len(units) > 0 and self.eos_idx == units[-1] or len(states.units.target) > self.max_len:
            tokens = self.model.decoder.dictionary.string([unit for unit in units if unit != DEFAULT_EOS])
            return [tokens.replace(BOW_PREFIX, ""), DEFAULT_EOS]

        return None

    def update_states_read(self, states):
        # Happens after a read action.
        if not states.finish_read():
            states.new_segment = True

    def prefix(self, states):
        """
        This method takes *states* as input, which stores the hypothesis generated at each time step
        in *states.chunks_hyp*, and returns the common prefix among the last *self.la_n* hypotheses
        without including the already displayed prefix *states.displayed*.
        """
        if states.finish_read() and len(states.chunks_hyp) > 0:
            displayed = len(states.displayed)
            return states.chunks_hyp[-1][displayed:]

        if len(states.chunks_hyp) < self.la_n:
            return []

        displayed = len(states.displayed)

        prefixes = [s[displayed:] for s in states.chunks_hyp[-self.la_n:]]
        common_pref = []
        for prefix in zip(*prefixes):
            prefix_candidate = prefix[0]
            if all(prefix_el == prefix_candidate for prefix_el in prefix) and prefix_candidate != self.eos_idx:
                common_pref.append(prefix_candidate)
            else:
                break

        return common_pref

    def _get_prefix(self, states):
        if len(states.displayed) > 1 and states.displayed[0] == self.eos_idx:
            return torch.LongTensor([states.displayed[1:]])
        elif len(states.displayed) > 0:
            return torch.LongTensor([states.displayed])
        else:
            return None

    def _predict(self, states):
        """
        This method takes *states* as input, which contains the audio input *states.units.source*, generates a
        translation hypothesis, and applies the *self.prefix()* method to obtain the common prefix among the
        previously generated hypotheses. It returns False if the prefix is empty, meaning that there is no
        common prefix among the generated hypotheses, and True otherwise.
        """
        states.new_segment = False

        sample = {
            'net_input': {
                'src_tokens': self.to_device(states.units.source.value.unsqueeze(0)),
                'src_lengths': self.to_device(torch.LongTensor([states.units.source.value.size(0)]))
            }
        }
        prefix_tokens = self._get_prefix(states)
        if self.prefix_token_idx:
            if prefix is not None:
                prefix_tokens = torch.cat(
                    (torch.LongTensor([[self.prefix_token_idx]]), prefix_tokens), dim=1)
            else:
                prefix_tokens = torch.LongTensor([[self.prefix_token_idx]])

        prefix_tokens = self.to_device(prefix_tokens) if prefix_tokens is not None else None
        hypos = self.task.inference_step(self.generator, self.model, sample, prefix_tokens=prefix_tokens)

        hypo = hypos[0][0]  # We consider only the most likely hypothesis
        hypo_tokens = hypo['tokens'].int().cpu()
        if self.prefix_token_idx:
            hypo_tokens = hypo_tokens[1:]

        states.chunks_hyp.append(hypo_tokens)
        common_pref = self.prefix(states)

        if len(common_pref) > 0:
            states.displayed.extend(common_pref)
            states.write = common_pref
            return True
        return False

    def policy(self, states):
        # Set a maximum to avoid possible loops by the system
        if len(states.units.target.value) > self.args.max_len:
            states.status['write'] = False

        if len(states.write) > 0:
            return WRITE_ACTION
        if states.new_segment and self._predict(states):
            return WRITE_ACTION
        if states.finish_read():
            # finish writing the hypo
            if self.prefix_token_idx and states.chunks_hyp[-1][0] == self.prefix_token_idx:
                states.chunks_hyp[-1] = states.chunks_hyp[-1][1:]
            states.write = states.chunks_hyp[-1][len(states.displayed):]
            return WRITE_ACTION
        return READ_ACTION

    def predict(self, states):
        if len(states.write) == 0:
            return self.eos_idx
        w = states.write[0]
        states.write = states.write[1:]
        return w
