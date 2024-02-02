#!/usr/bin/python3
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
import argparse
import os

import numpy as np
import logging

from numpy.lib.stride_tricks import sliding_window_view

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
)
LOGGER = logging.getLogger(__file__)


class GeneratedTokens:
    def __init__(self, token_string: str):
        self.tokens = token_string.strip().split(" ")

    def __len__(self):
        return len(self.tokens)

    def eob_indexes(self):
        return [idx for idx, x in enumerate(self.tokens) if x == "<eob>"]

    def assert_ends_with_eob(self):
        if self.tokens[-1] != "<eob>":
            LOGGER.warning(f"Line does not end with eob: {self.tokens}.")
            self.tokens[-1] = "<eob>"


class AttentionMatrixProcessor:
    def __init__(self, attention_string: str):
        self.attention = np.array([
            list(map(float, attn_row.split(","))) for attn_row in attention_string.split(" ")])

    def __len__(self):
        return len(self.attention)

    def __str__(self):
        return " ".join(",".join(str(x) for x in row) for row in self.attention)

    def remove_eos(self):
        self.attention = self.attention[:-1, :]

    def audio_len(self):
        return self.attention.shape[1]

    def remove_lang(self):
        self.attention = self.attention[1:, :]

    def remove_last_frame(self):
        self.attention = self.attention[:, :-1]

    def normalize(self):
        self.std_normalize()

    def std_normalize(self):
        """
        Standard normalization with normal distribution.
        """
        std = self.attention.std(axis=0)
        std[std == 0.] = 1.0
        mean = self.attention.mean(axis=0)
        self.attention = self.attention - mean
        self.attention = self.attention / std

    def aligns(self, boundaries_indexes):
        raise NotImplementedError("Subclasses of AttentionMatrixProcessor should implement aligns")


class CustomAttentionAligner(AttentionMatrixProcessor):
    """
    Custom method specifically designed by FBK to determine block boundaries,
    trying to maximize the value of the attention area of corresponding text and audio.
    """
    def normalize(self):
        self.std_normalize()
        # set all attention elements below the mean to a small negative value
        # to discourage adding useless areas, while avoiding penalizing too much
        # with very negative values that may depend only on how peaky the attention is
        self.attention[self.attention < 0] = -0.01

    def _score(self, start_text, end_text, start_time, end_time):
        return self.attention[start_text:end_text, start_time:end_time].sum()

    def aligns(self, boundaries_indexes):
        """
        Iteratively determines the block boundaries by choosing the splitting point that
        maximizes the area of the first block and the rest of the text, then the second
        block and the rest of the text and so on.
        """
        start_text_i = 0
        start_time_i = 0
        splitting_time_idxs = []
        text_len, audio_len = self.attention.shape
        idxs_to_align = len(boundaries_indexes)
        for boundary_idx in boundaries_indexes:
            best_candidate_score = float("-inf")
            best_candidate_idx = None
            end_text_i = boundary_idx + 1
            max_end_time = audio_len - idxs_to_align  # leave at least one frame per remaining block
            for cand_time_i in range(start_time_i + 1, audio_len - idxs_to_align):
                cand_score = self._score(start_text_i, end_text_i, start_time_i, cand_time_i) \
                             + self._score(end_text_i, text_len, cand_time_i, max_end_time)
                if cand_score > best_candidate_score:
                    best_candidate_idx = cand_time_i
                    best_candidate_score = cand_score
            if best_candidate_idx is None:
                # this can happen in error conditions e.g. the text generated is longer than audio.
                best_candidate_idx = audio_len - idxs_to_align
            splitting_time_idxs.append(best_candidate_idx)
            idxs_to_align -= 1
            start_time_i = best_candidate_idx
            start_text_i = boundary_idx
        return splitting_time_idxs


class CustomForcedEndAttentionAligner(CustomAttentionAligner):
    """
    The current method does not properly estimate the end time of the last eob.
    As a workaround, this forces the last eob to terminate at the end of the audio.
    """
    def aligns(self, boundaries_indexes):
        splitting_time_idxs = super().aligns(boundaries_indexes)
        if len(splitting_time_idxs) > 0:
            splitting_time_idxs[-1] = self.audio_len()
        return splitting_time_idxs


class DTWMedianFilterAttentionAligner(AttentionMatrixProcessor):
    """
    Alignment method inspired to the token alignment used in the `HuggingFace Whisper implementation
    <https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/whisper/modeling_whisper.py>`_
    and then considers the time assigned to the <eob>s.

    It implements `"An improved dynamic time warping algorithm employing nonlinear median filtering"
    <https://ieeexplore.ieee.org/document/6089967>`_.
    """
    def _median_filter(self, filter_width: int):
        """
        Applies a median filter of width `filter_width` along the last dimension of the input.
        """
        if filter_width <= 0 or filter_width % 2 != 1:
            raise ValueError("`filter_width` should be an odd number")

        pad_width = filter_width // 2
        if self.attention.shape[-1] <= pad_width:
            return

        # Pad the left and right edges.
        inputs = np.pad(self.attention, ((0, 0), (pad_width, pad_width)), mode="reflect")
        self.attention = np.median(sliding_window_view(inputs, filter_width, axis=-1), axis=-1)

    def normalize(self):
        self.std_normalize()
        self._median_filter(7)

    def _dynamic_time_warping(self, boundaries):
        """
        Uses the negated attention as similarity between the input audio and the output tokens.
        Returns two lists of the same size that describe the alignment path, where at each position
        either the text indices or the time indices is increased (or both).
        Used to generate token-level timestamps.
        """
        output_length, input_length = self.attention.shape
        cost = np.ones((output_length + 1, input_length + 1), dtype=np.float32) * np.inf
        trace = -np.ones((output_length + 1, input_length + 1), dtype=np.float32)

        cost[0, 0] = 0
        for j in range(1, input_length + 1):
            for i in range(1, output_length + 1):
                c0 = cost[i - 1, j - 1]
                c1 = cost[i - 1, j]
                c2 = cost[i, j - 1]

                if c0 < c1 and c0 < c2:
                    c, t = c0, 0
                elif c1 < c0 and c1 < c2:
                    c, t = c1, 1
                else:
                    c, t = c2, 2

                cost[i, j] = - self.attention[i - 1, j - 1] + c
                trace[i, j] = t

        # backtrace
        i = trace.shape[0] - 1
        j = trace.shape[1] - 1
        # ensure at least one time step per block
        trace[2:, 1] = 1
        for i_b, b_idx in enumerate(boundaries):
            trace[b_idx + 1:, i_b + 2] = 1
            trace[b_idx + 1, :] = 0

        # force finishing the time dimension
        trace[0, :] = 2
        # force finishing the text dimension
        trace[:, 0] = 1

        text_indices = []
        time_indices = []
        while i > 0 or j > 0:
            text_indices.append(i - 1)
            time_indices.append(j - 1)
            if trace[i, j] == 0:
                i -= 1
                j -= 1
            elif trace[i, j] == 1:
                i -= 1
            elif trace[i, j] == 2:
                j -= 1
            else:
                raise RuntimeError(
                    f"Internal error in dynamic time warping. Unexpected trace[{i}, {j}]."
                    f" Please file a bug report.")

        text_indices = np.array(text_indices)[::-1]
        time_indices = np.array(time_indices)[::-1]
        return text_indices, time_indices

    def aligns(self, boundaries_indexes):
        text_indices, time_indices = self._dynamic_time_warping(boundaries_indexes)
        # detect array positions where the text changes to get token-level timestamps
        jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)
        return time_indices[jumps][boundaries_indexes]


class AttentionAlignerArgparse(argparse.Action):
    AVAILABLE_ALIGNERS = {
        "custom": CustomAttentionAligner,
        "custom-forceend": CustomForcedEndAttentionAligner,
        "dtw-medianf": DTWMedianFilterAttentionAligner,
    }

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, self.AVAILABLE_ALIGNERS[values])


class CTCCompressionSizes:
    """
    When the input audio is dynamically compressed by the CTC compression
    this keeps track of how many time steps correspond to a given encoder output.
    """
    def __init__(self, ctc_string: str):
        self.ctc_sizes = list(map(float, ctc_string.split(",")))

    def __len__(self):
        return len(self.ctc_sizes)

    def get_real_times_ms(self, idxs):
        times = []
        t_idx = 0
        t_acc = 0
        for idx in sorted(idxs):
            while t_idx < idx:
                t_acc += self.ctc_sizes[t_idx] * 40
                t_idx += 1
            times.append(t_acc)
        return times


def main(args):
    if args.debug_plotting:
        import matplotlib.pyplot as plt
    generated_tokens = {}
    attentions = {}
    ctc_compression_sizes = {}
    with open(args.fairseq_output, 'r') as fd:
        for line in fd:
            if not (line.startswith('H-') or line.startswith('A-') or line.startswith('CTC-')):
                continue
            line_parts = line.split('\t')
            sample_id = int(line_parts[0].split('-')[-1])
            if line.startswith('H-'):
                generated_tokens[sample_id] = GeneratedTokens(line_parts[-1])
            if line.startswith('A-'):
                attentions[sample_id] = args.alignment_operator(line_parts[-1])
            if line.startswith('CTC-'):
                ctc_compression_sizes[sample_id] = CTCCompressionSizes(line_parts[-1])

    for i in range(len(generated_tokens)):
        if len(generated_tokens[i]) == len(attentions[i]) - 1:
            # we need to remove the last row of the attention matrix
            # which corresponds to the eos token
            attentions[i].remove_eos()
        elif len(generated_tokens[i]) == len(attentions[i]) - 2:
            # multilingual generation, we need to remove the last
            # row (which corresponds to the eos token) and the first
            # row (which corresponds to the lang token) of the attention matrix
            attentions[i].remove_eos()
            attentions[i].remove_lang()
        else:
            raise ValueError(
                f"Impossible to map {len(generated_tokens[i])} tokens to {len(attentions[i])} "
                "attention rows")
        assert len(generated_tokens[i]) == len(attentions[i])
        assert attentions[i].audio_len() == len(ctc_compression_sizes[i])
        if args.remove_last_frame:
            attentions[i].remove_last_frame()
        attentions[i].normalize()
        generated_tokens[i].assert_ends_with_eob()
        eob_indexes = generated_tokens[i].eob_indexes()
        eob_time_indexes = attentions[i].aligns(eob_indexes)
        if len(eob_time_indexes) == 0:
            eob_time_indexes = [attentions[i].audio_len()]
        eob_times = ctc_compression_sizes[i].get_real_times_ms(eob_time_indexes)
        time_strings = []
        for idx, end_time in enumerate(eob_times):
            if idx == 0:
                start_time = 0.
            else:
                start_time = eob_times[idx - 1]
            if start_time == end_time:
                LOGGER.error(
                    f"The {idx}-th block of line {i} is assigned zero length "
                    f"({float(start_time)/1000}-{float(end_time)/1000})! This is a bug, so please "
                    "report the issue with this error message, including the boundary token ids "
                    f"({eob_indexes}) and the attention below for reproducibility: \n"
                    f"{attentions[i]}")
            time_strings.append(f"{float(start_time)/1000}-{float(end_time)/1000}")
        if args.debug_plotting:
            plt.rc('xtick', labelsize=8)
            plt.rc('ytick', labelsize=8)
            fig = plt.figure(figsize=(60, 10))
            ax = fig.add_subplot(111)
            ax.matshow(attentions[i].attention, cmap='hot', interpolation='nearest', vmin=0., vmax=1.)
            ax.set_yticks(np.arange(len(generated_tokens[i])), generated_tokens[i].tokens)
            plt.vlines(eob_time_indexes, ymin=0, ymax=eob_indexes, colors="red")
            plt.hlines(eob_indexes, xmin=0, xmax=eob_time_indexes, colors="red")
            fig.tight_layout()
            plt.show()
        print(" ".join(time_strings))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Attention-based timestamp estimation from fairseq output")
    parser.add_argument('--fairseq-output', type=str, required=True,
                        help="output of the fairseq generate with texts and attention weights")
    parser.add_argument('--alignment-operator',
                        action=AttentionAlignerArgparse,
                        choices=AttentionAlignerArgparse.AVAILABLE_ALIGNERS.keys(),
                        default=AttentionAlignerArgparse.AVAILABLE_ALIGNERS['custom-forceend'],
                        help="method to use to perform alignments")
    parser.add_argument('--remove-last-frame', action='store_true', default=False,
                        help="if set, last token is removed before computing alignments")
    parser.add_argument('--debug-plotting', action='store_true', default=False,
                        help="if set, every attention is plotted to debug what is happening")
    main(parser.parse_args())
