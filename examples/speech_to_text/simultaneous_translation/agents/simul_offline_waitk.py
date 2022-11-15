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

import itertools
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.compliance.kaldi as kaldi
import yaml

from fairseq import checkpoint_utils, tasks
from fairseq.file_io import PathManager
from fairseq.utils import import_user_module

try:
    from simuleval import READ_ACTION, WRITE_ACTION, DEFAULT_EOS
    from simuleval.agents import SpeechAgent
    from simuleval.states import ListEntry, SpeechStates
except ImportError:
    print("Please install simuleval 'pip install simuleval'")
    raise ImportError

SHIFT_SIZE = 10
WINDOW_SIZE = 25
SAMPLE_RATE = 16000
FEATURE_DIM = 80
BOW_PREFIX = "\u2581"


class OnlineFeatureExtractor:
    """
    Extract speech feature on the fly.
    """

    def __init__(self, args):
        self.shift_size = args.shift_size
        self.window_size = args.window_size
        assert self.window_size >= self.shift_size

        self.sample_rate = args.sample_rate
        self.feature_dim = args.feature_dim
        self.num_samples_per_shift = self.shift_size * self.sample_rate / 1000
        self.num_samples_per_window = self.window_size * self.sample_rate / 1000
        self.len_ms_to_samples = (self.window_size - self.shift_size) * self.sample_rate / 1000
        self.previous_residual_samples = []
        self.global_cmvn = args.global_cmvn

    def clear_cache(self):
        self.previous_residual_samples = []

    def __call__(self, new_samples):
        # samples is composed by new received samples + residuals
        # to correctly compute audio features through shift and window
        samples = self.previous_residual_samples + new_samples
        if len(samples) < int(self.num_samples_per_window):
            self.previous_residual_samples = samples
            return

        # num_frames is the number of frames from the new segment
        num_frames = math.floor(
            (len(samples) - self.len_ms_to_samples)
            / self.num_samples_per_shift
        )

        # the number of frames used for feature extraction
        # including some part of the previous segment
        effective_num_samples = int(
            (num_frames * self.num_samples_per_shift)
            + self.len_ms_to_samples
        )

        input_samples = samples[:effective_num_samples]
        self.previous_residual_samples = samples[num_frames * int(self.num_samples_per_shift):]

        output = kaldi.fbank(
            torch.FloatTensor(input_samples).unsqueeze(0),
            num_mel_bins=self.feature_dim,
            frame_length=self.window_size,
            frame_shift=self.shift_size,
        )
        return self.transform(output)

    def transform(self, x):
        if self.global_cmvn is None:
            return x
        return (x - self.global_cmvn["mean"]) / self.global_cmvn["std"]


class TensorListEntry(ListEntry):
    """
    Data structure to store a list of tensor.
    """

    def append(self, value):
        if len(self.value) == 0:
            self.value = value
            return
        self.value = torch.cat([self.value] + [value], dim=0)

    def info(self):
        return {
            "type": str(self.new_value_type),
            "length": len(self),
            "value": "" if type(self.value) is list else self.value.size(),
        }


class FairseqSimulSTAgent(SpeechAgent):
    speech_segment_size = 40  # in ms, 4 pooling ratio * 10 ms step size

    def __init__(self, args):
        super().__init__(args)
        self.eos = DEFAULT_EOS
        self.gpu = getattr(args, "gpu", False)
        self.args = args

        self.load_model_vocab(args)

        if getattr(
                self.model.decoder.layers[0].encoder_attn,
                'pre_decision_ratio',
                None
        ) is not None:
            self.speech_segment_size *= (
                self.model.decoder.layers[0].encoder_attn.pre_decision_ratio
            )

        self.speech_segment_size *= args.speech_segment_factor

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

    def build_states(self, args, client, sentence_id):
        # Initialize states here, for example add customized entry to states
        # This function will be called at beginning of every new sentence
        states = SpeechStates(args, client, sentence_id, self)
        self.initialize_states(states)
        return states

    def to_device(self, tensor):
        if self.gpu:
            return tensor.cuda()
        else:
            return tensor.cpu()

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--model-path', type=str, required=True,
                            help='path to your pretrained model.')
        parser.add_argument("--data-bin", type=str, required=True,
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
        parser.add_argument("--force-finish", default=False, action="store_true",
                            help="Force the model to finish the hypothsis if the source is not finished")
        parser.add_argument("--shift-size", type=int, default=SHIFT_SIZE,
                            help="Shift size of feature extraction window.")
        parser.add_argument("--window-size", type=int, default=WINDOW_SIZE,
                            help="Window size of feature extraction window.")
        parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE,
                            help="Sample rate")
        parser.add_argument("--feature-dim", type=int, default=FEATURE_DIM,
                            help="Acoustic feature dimension.")
        parser.add_argument("--waitk", type=int, default=None,
                            help="Wait k lagging value for test.")
        parser.add_argument("--speech-segment-factor", type=int, default=1,
                            help="Factor multiplied by speech segment size of 40ms to obtain the final speech segment "
                                 "size used in the adaptive module.")
        parser.add_argument("--adaptive-segmentation", default=False, action="store_true",
                            help="Whether to enable CTC prediction (greedy) to identify the number of words contained "
                                 "in the speech.")
        parser.add_argument("--vocabulary-type", type=str, choices=["sentencepiece", "subwordnmt"], default=None,
                            help="If adaptive segmentation is used, the vocabulary type has to be specified")

        # fmt: on
        return parser

    def load_model_vocab(self, args):

        filename = args.model_path
        if not os.path.exists(filename):
            raise IOError("Model file not found: {}".format(filename))

        state = checkpoint_utils.load_checkpoint_to_cpu(filename)

        task_args = state["cfg"]["task"]
        task_args.data = args.data_bin

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
        self.model.share_memory()

        self.generator = self.task.build_generator(
            [self.model], state["cfg"]["generation"]
        )

        if self.gpu:
            self.model.cuda()

        # Check vocabulary type
        if self.args.adaptive_segmentation:
            assert self.args.vocabulary_type is not None

        self.tgt_dict = self.model.decoder.dictionary
        self.src_dict = self.model.encoder.dictionary
        self.ctc_blank_idx = self.model.encoder.dictionary.index("<ctc_blank>")

    def initialize_states(self, states):
        self.feature_extractor.clear_cache()
        states.units.source = TensorListEntry()
        states.units.target = ListEntry()
        states.incremental_states = dict()
        states.prev_toks = []
        states.to_predict = []
        states.n_predicted_words = 0
        states.n_audio_words = 0
        states.full_pred = None

    def segment_to_units(self, segment, states):
        # Convert speech samples to features
        features = self.feature_extractor(segment)
        if features is not None:
            return [features]
        else:
            return []

    def units_to_segment(self, units, states):
        # Merge sub word to full word.
        if self.model.decoder.dictionary.eos() == units[0]:
            return DEFAULT_EOS

        segment = []
        if None in units.value:
            units.value.remove(None)

        for index in units:
            if index is None:
                units.pop()
            token = self.model.decoder.dictionary.string([index])
            if token.startswith(BOW_PREFIX):
                if len(segment) == 0:
                    segment += [token.replace(BOW_PREFIX, "")]
                else:
                    for j in range(len(segment)):
                        units.pop()

                    string_to_return = ["".join(segment)]

                    if self.model.decoder.dictionary.eos() == units[0]:
                        string_to_return += [DEFAULT_EOS]

                    return string_to_return
            else:
                segment += [token.replace(BOW_PREFIX, "")]

        if (
                len(units) > 0
                and self.model.decoder.dictionary.eos() == units[-1]
                or len(states.units.target) > self.max_len
        ):
            tokens = [self.model.decoder.dictionary.string([unit]) for unit in units]
            return ["".join(tokens).replace(BOW_PREFIX, "")] + [DEFAULT_EOS]

        return None

    def update_model_encoder(self, states):
        if len(states.units.source) == 0:
            return
        src_indices = self.to_device(
            states.units.source.value.unsqueeze(0)
        )
        src_lengths = self.to_device(
            torch.LongTensor([states.units.source.value.size(0)])
        )

        states.encoder_states = self.model.encoder(src_indices, src_lengths)
        states.n_audio_words = self.get_audio_words(states)
        torch.cuda.empty_cache()

    def update_states_read(self, states):
        # Happens after a read action.
        if not states.finish_read():
            self.update_model_encoder(states)

    def get_audio_words(self, states):
        ctc_pred_greedy = F.log_softmax(states.encoder_states["ctc_out"], dim=-1).transpose(0, 1).max(dim=-1)[1]
        ctc_pred_greedy_unique = [k for k, _ in itertools.groupby(ctc_pred_greedy.tolist()[0])]
        audio_words = self.src_dict.string(
            [tok for tok in ctc_pred_greedy_unique if tok != self.ctc_blank_idx],
            bpe_symbol=self.args.vocabulary_type
        )
        return len(audio_words.split(" ")) if audio_words != "" else 0

    def _get_prefix(self, states):
        if states.prev_toks:
            return torch.LongTensor(states.prev_toks).unsqueeze(0)
        else:
            return None

    def _select_words(self, new_hypo, selected_n_words):
        selected_idxs = []
        curr_n_words = 0
        for idx in new_hypo:
            if self.tgt_dict.string([idx]).startswith(BOW_PREFIX):
                curr_n_words += 1
                if curr_n_words > selected_n_words:
                    break
            selected_idxs.append(idx)
        return selected_idxs

    def _predict(self, states):
        sample = {
            'net_input': {
                'src_tokens': self.to_device(states.units.source.value.unsqueeze(0)),
                'src_lengths': self.to_device(torch.LongTensor([states.units.source.value.size(0)]))
            }
        }

        prefix_tokens = self.to_device(self._get_prefix(states)) if self._get_prefix(states) is not None else None
        with torch.no_grad():
            hypos = self.generator.generate(
                self.model, sample, prefix_tokens=prefix_tokens, pre_computed_encoder_outs=[states.encoder_states]
            )
        hypo = hypos[0][0]
        hypo_tokens = hypo['tokens'].int().cpu()
        return hypo_tokens

    def waitk_prediction(self, states):
        hypo_tokens = self._predict(states)

        states.full_pred = hypo_tokens
        new_hypo = hypo_tokens[len(states.prev_toks):]
        selected_n_words = states.n_audio_words - (states.n_predicted_words + self.args.waitk)
        states.n_predicted_words += selected_n_words
        selected_idxs = self._select_words(new_hypo, selected_n_words)

        torch.cuda.empty_cache()

        if selected_idxs:
            states.to_predict = selected_idxs
            states.prev_toks += selected_idxs
            return True
        else:
            states.prev_toks += selected_idxs
            return False

    def policy(self, states):
        if not getattr(states, "encoder_states", None):
            return READ_ACTION

        if len(states.to_predict) > 0:
            return WRITE_ACTION

        # Set a maximum to avoid possible loops by the system
        if states.n_predicted_words > self.args.max_len:
            states.status['write'] = False

        # Number of words for the wait-k policy
        n_words = states.n_predicted_words + self.args.waitk

        action = 0
        # If finish_read, write the remaining output
        if states.finish_read():
            if states.full_pred is None:
                states.full_pred = self._predict(states)
            final_hypo = states.full_pred[len(states.prev_toks):]
            states.to_predict = final_hypo
            return WRITE_ACTION

        if self.args.adaptive_segmentation:
            # Adaptive segmentation based on the CTC prediction (greedy)
            if states.n_audio_words > n_words:
                action = 1
        else:
            # Fixed segmentation
            # Take an action based on the number of encoder states
            if states.encoder_states["ctc_lengths"].item() // self.args.speech_segment_factor > n_words:
                action = 1

        if action == 0:
            return READ_ACTION
        else:
            if self.waitk_prediction(states):
                return WRITE_ACTION
            else:
                return READ_ACTION

    def predict(self, states):
        idx = states.to_predict[0]
        states.to_predict = states.to_predict[1:]
        return idx
