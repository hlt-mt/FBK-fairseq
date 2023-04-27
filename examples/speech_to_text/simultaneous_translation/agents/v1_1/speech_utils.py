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

import torch
import torchaudio.compliance.kaldi as kaldi
import math

from simuleval.agents import AgentStates
from simuleval.data.segments import EmptySegment, TextSegment

from examples.speech_to_text.simultaneous_translation.agents.speech_utils import OnlineFeatureExtractor


class SpeechStates(AgentStates):
    def __init__(self, agent):
        super().__init__()
        self.agent = agent

    def update_source(self, segment):
        # Executed when a new input speech segment is received
        if isinstance(segment, EmptySegment):
            return
        # Extract filterbanks from the input content
        features = self.agent.get_filterbank(segment.content)
        # Add the new input to the already received content
        self.source = [torch.cat(self.source + features, dim=0)]
        # Update the model states after the new input content has been received
        self.agent.update_states_read(self)
        self.source_finished = segment.finished

    def update_target(self, segment):
        self.target_finished = segment.finished
        # If the model is still producing the output (i.e., <eos> has not been
        # emitted yet), store the output text segment produced at each time step
        if not self.target_finished:
            if isinstance(segment, TextSegment):
                self.target.append(segment.content)
            else:
                self.target += segment.content


class OnlineFeatureExtractorV1_1(OnlineFeatureExtractor):
    def __call__(self, new_samples):
        # samples is composed by new received samples + residuals
        # to correctly compute audio features through shift and window
        samples = self.previous_residual_samples + new_samples
        if len(samples) < int(self.num_samples_per_window):
            self.previous_residual_samples = samples
            return

        # num_frames is the number of frames from the new segment
        num_frames = math.floor(
            (len(samples) - self.len_ms_to_samples) / self.num_samples_per_shift)

        # the number of frames used for feature extraction
        # including some part of the previous segment
        effective_num_samples = int(
            (num_frames * self.num_samples_per_shift) + self.len_ms_to_samples)

        input_samples = torch.FloatTensor(samples[:effective_num_samples]).unsqueeze(0)
        # Kaldi compliance: 16-bit signed integers
        input_samples = input_samples * (2 ** 15)
        self.previous_residual_samples = samples[num_frames * int(self.num_samples_per_shift):]

        output = kaldi.fbank(
            input_samples,
            num_mel_bins=self.feature_dim,
            frame_length=self.window_size,
            frame_shift=self.shift_size)
        return self.transform(output)
