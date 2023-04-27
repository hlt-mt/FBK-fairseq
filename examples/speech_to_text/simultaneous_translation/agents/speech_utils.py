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

import math
import torch
import torchaudio.compliance.kaldi as kaldi

DEFAULT_EOS = '</s>'
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
            / self.num_samples_per_shift)

        # the number of frames used for feature extraction
        # including some part of the previous segment
        effective_num_samples = int(
            (num_frames * self.num_samples_per_shift)
            + self.len_ms_to_samples)

        input_samples = samples[:effective_num_samples]
        self.previous_residual_samples = samples[num_frames * int(self.num_samples_per_shift):]

        output = kaldi.fbank(
            torch.FloatTensor(input_samples).unsqueeze(0),
            num_mel_bins=self.feature_dim,
            frame_length=self.window_size,
            frame_shift=self.shift_size)
        return self.transform(output)

    def transform(self, x):
        if self.global_cmvn is None:
            return x
        return (x - self.global_cmvn["mean"]) / self.global_cmvn["std"]
