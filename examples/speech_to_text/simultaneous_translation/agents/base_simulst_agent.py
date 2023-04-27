# Copyright 2022 FBK

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import json
import os

import numpy as np
import torch
import yaml

from examples.speech_to_text.simultaneous_translation.agents.speech_utils import DEFAULT_EOS, \
    OnlineFeatureExtractor, SHIFT_SIZE, WINDOW_SIZE, SAMPLE_RATE, FEATURE_DIM
from fairseq import checkpoint_utils, tasks
from fairseq.data.audio.speech_to_text_dataset import SpeechToTextDataset
from fairseq.file_io import PathManager
from fairseq.utils import import_user_module


class BaseSimulSTAgent:
    """
    Base agent for Simultaneous Speech Translation.
    It includes generic methods to:
    - parse the command line arguments;
    - load both model, task, and generator (*load_model_vocab*);
    - generate the hypothesis through *generate_hypothesis*.
    """
    def __init__(self, args):
        self.eos = DEFAULT_EOS
        self.eos_idx = None
        self.prefix_token_idx = None
        self.gpu = getattr(args, "gpu", False)

        self.args = args

        self.load_model_vocab()

        args.global_cmvn = None
        if args.config:
            with open(os.path.join(args.data_bin, args.config), "r") as f:
                config = yaml.load(f, Loader=yaml.BaseLoader)

            if "global_cmvn" in config:
                args.global_cmvn = np.load(config["global_cmvn"]["stats_npz_path"])

        if args.global_stats:
            with PathManager.open(args.global_stats, "r") as f:
                global_cmvn = json.loads(f.read())
                self.global_cmvn = {
                    "mean": torch.from_numpy(global_cmvn["mean"]),
                    "std": torch.from_numpy(global_cmvn["stddev"])
                }

        self.feature_extractor = OnlineFeatureExtractor(args)
        self.max_len = args.max_len

        torch.set_grad_enabled(False)

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--model-path', type=str, required=True,
                            help='path to your pretrained model.')
        parser.add_argument("--data-bin", type=str, default=None,
                            help="Path of data binary")
        parser.add_argument("--config", type=str, default=None,
                            help="Path to config yaml file")
        parser.add_argument("--global-stats", type=str, default=None,
                            help="Path to json file containing cmvn stats")
        parser.add_argument("--tgt-splitter-type", type=str, default="SentencePiece",
                            help="Subword splitter type for target text")
        parser.add_argument("--tgt-splitter-path", type=str, default=None,
                            help="Subword splitter model path for target text")
        parser.add_argument("--user-dir", type=str, default="examples/simultaneous_translation",
                            help="User directory for simultaneous translation")
        parser.add_argument("--max-len", type=int, default=200,
                            help="Max length of translation")
        parser.add_argument("--shift-size", type=int, default=SHIFT_SIZE,
                            help="Shift size of feature extraction window.")
        parser.add_argument("--window-size", type=int, default=WINDOW_SIZE,
                            help="Window size of feature extraction window.")
        parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE,
                            help="Sample rate")
        parser.add_argument("--feature-dim", type=int, default=FEATURE_DIM,
                            help="Acoustic feature dimension.")
        parser.add_argument("--prefix-size", default=0, type=int,
                            help="Initialize generation by target prefix of given length.")
        parser.add_argument("--prefix-token", default="", type=str,
                            help="Target prefix token.")
        parser.add_argument("--beam", type=int, default=5,
                            help="Beam size.")
        parser.add_argument("--no-repeat-ngram-size", type=int, default=0,
                            help="Same parameter of the the fairseq-generate.")
        parser.add_argument("--speech-segment-factor", type=int, default=1,
                            help="Factor to be multiplied by the speech segment size to obtain the "
                                 "final speech segment size.")
        # fmt: on
        return parser

    def build_model(self, state):
        state["cfg"]["model"].load_pretrained_encoder_from = None
        state["cfg"]["model"].load_pretrained_decoder_from = None
        self.model = self.task.build_model(state["cfg"]["model"])
        self.model.load_state_dict(state["model"], strict=True)
        self.model.eval()
        # share model tensors to let SimulEval processes to interact and exchange information and data
        self.model.share_memory()

    def load_model_vocab(self):
        filename = self.args.model_path
        if not os.path.exists(filename):
            raise IOError("Model file not found: {}".format(filename))

        state = checkpoint_utils.load_checkpoint_to_cpu(filename)

        task_args = state["cfg"]["task"]
        if self.args.data_bin is not None:
            task_args.data = self.args.data_bin

        if self.args.config is not None:
            task_args.config_yaml = self.args.config

        import_user_module(state["cfg"].common)

        self.task = tasks.setup_task(task_args)

        self.build_model(state)

        state["cfg"]["generation"]["beam"] = self.args.beam
        state["cfg"]["generation"]["no_repeat_ngram_size"] = self.args.no_repeat_ngram_size
        state["cfg"]["generation"]["prefix_size"] = self.args.prefix_size
        self.generator = self.task.build_generator([self.model], state["cfg"]["generation"])

        if self.gpu:
            self.model.cuda()

        self.tgtdict = self.model.decoder.dictionary
        self.srcdict = self.model.encoder.dictionary

        if self.args.prefix_token != "":
            lang_tag = SpeechToTextDataset.LANG_TAG_TEMPLATE.format(self.args.prefix_token)
            self.prefix_token_idx = self.tgtdict.index(lang_tag)

        self.eos_idx = self.tgtdict.index(DEFAULT_EOS)

    @staticmethod
    def _get_prefix_len(prefix):
        return len(prefix[0]) if prefix is not None else 0

    def _generate_hypothesis(self, sample, prefix_tokens, states):
        # Hypothesis generation
        with torch.no_grad():
            hypos = self.generator.generate(
                self.model, sample, prefix_tokens=prefix_tokens,
                pre_computed_encoder_outs=[states.encoder_states]
            )
        return hypos

    def _emit_remaining_tokens(self, states):
        """
        It implements the actions to be performed when
        the input has been completely received.
        """
        pass

    def _policy(self, states):
        """
        It implements the logic of the policy that determines whether to
        read more input or emit a full or partial translation.
        """
        pass
