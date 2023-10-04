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
from typing import Tuple
import numpy as np

import torch
from torch import Tensor

from fairseq.data.audio.feature_transforms import AudioFeatureTransform, register_audio_feature_transform


@register_audio_feature_transform("vtlp")
class VTLP(AudioFeatureTransform):
    """
    Vocal tract length perturbation (original paper:
    http://www.cs.toronto.edu/~hinton/absps/perturb.pdf).
    The code is inspired by the implementations in
    https://github.com/waveletdeboshir/speechaugs/blob/master/speechaugs/speechaugs.py#L479
    and in https://github.com/makcedward/nlpaug/blob/master/nlpaug/model/audio/vtlp.py
    It receives as input the waveform (1 channel) and returns the perturbed waveform.
    Parameters:
        - sr: sample rate
        - warp_factor: range in which the warping factor is sampled
        - sampling_type: type of distribution from which to sample the warping factor
                        (`normal` or `uniform`)
        - boundary_freq: a boundary frequency up to which the warping function is applied
    """

    def __init__(
            self,
            sr: int = 16000,
            warp_factor_boundaries: Tuple[float, float] = (0.9, 1.1),
            sampling_type: str = "random",
            boundary_freq: int = 4800):
        assert warp_factor_boundaries[0] < warp_factor_boundaries[1], \
            "The range of warping factors is not correct; the minimum should be provided first."
        assert boundary_freq <= sr // 2, \
            "The boundary frequency should be lower than the nyquist frequency."
        assert sampling_type in ["random", "uniform"], \
            "Invalid sampling type. Only 'random' and 'uniform' are accepted"
        self.sr = sr
        self.nyquist_f = self.sr / 2
        self.min_warp_factor = warp_factor_boundaries[0]
        self.max_warp_factor = warp_factor_boundaries[1]
        self.sampling_type = sampling_type
        self.boundary_freq = boundary_freq

    @staticmethod
    def get_scale_factor(
            sampling_type: str,
            min_scale_factor: float,
            max_scale_factor: float) -> float:
        """Returns the warping factor."""
        # Unlike the implementation in
        # https://github.com/waveletdeboshir/speechaugs/blob/master/speechaugs/speechaugs.py#L479
        # we add the possibility (default) to generate the scaling factor  from a normal distribution
        # centered at 1, with a standard deviation of 0.1, as per the original paper
        if sampling_type == "random":
            scale_factor = np.random.normal(1, (max_scale_factor - min_scale_factor) / 2)
            scale_factor = min(max(scale_factor, min_scale_factor), max_scale_factor)
        else:
            scale_factor = np.random.uniform(min_scale_factor, max_scale_factor)
        return scale_factor

    def get_new_freqs(self, n_fft: int):
        """
        Returns the warped values of the frequencies values of each bin applying equation 2 of
        the original paper.
        """
        new_freqs = []
        alpha = self.get_scale_factor(self.sampling_type, self.min_warp_factor, self.max_warp_factor)
        bound = self.boundary_freq / alpha if alpha > 1 else self.boundary_freq
        freqs = np.linspace(0, 1, (n_fft // 2) + 1) * self.nyquist_f
        for fr in freqs:
            # Based on the original paper, the condition should be "fr <= bound",
            # but the existing open-source implementations used "fr < bound".
            # Here we follow previous implementations.
            if fr < bound:
                new_freq = fr * alpha
            else:
                delta_freq = self.nyquist_f - self.boundary_freq * min(alpha, 1)
                adjusted_freq = delta_freq / (self.nyquist_f - bound) * (self.nyquist_f - fr)
                new_freq = self.nyquist_f - adjusted_freq
            new_freqs.append(new_freq)
        return torch.tensor(new_freqs)

    def get_new_spec(self, spectrogram: Tensor, new_freqs: Tensor, n_fft: int) -> Tensor:
        """
        Returns the new warped spectrogram using the warped values of the frequencies values
        of each bin.
        """
        n_freqs, n_time = spectrogram.shape
        new_spec = torch.zeros(n_freqs, n_time, dtype=spectrogram[0, 0].dtype)
        for i in range(n_freqs):
            if 0 < i < n_freqs - 1:  # skip the first and last elements
                warp_up = new_freqs[i] - torch.floor(new_freqs[i])
                warp_down = 1 - warp_up
                pos = int(torch.floor(new_freqs[i]) * n_fft / self.sr)
                new_spec[pos, :] += warp_down * spectrogram[i, :]
                new_spec[pos + 1, :] += warp_up * spectrogram[i, :]
            else:
                new_spec[i, :] += spectrogram[i, :]
        return new_spec

    def __call__(self, waveform):
        wav_torch = torch.from_numpy(waveform)
        assert wav_torch.shape[0] == 1, \
            'waveform has more than 1 channel. It must have just 1 channel.'
        wav_torch = wav_torch.clone()
        n_fft = 512
        spectrogram = torch.stft(wav_torch.squeeze(0), n_fft=n_fft, return_complex=True)
        new_freqs = self.get_new_freqs(n_fft)
        new_spec = self.get_new_spec(spectrogram, new_freqs, n_fft)
        wav_perturbed = torch.istft(new_spec, n_fft)
        return wav_perturbed.unsqueeze(0).numpy()

    @classmethod
    def from_config_dict(cls, config=None):
        _config = {} if config is None else config
        return VTLP(
            _config.get("sampling_rate", 16000),
            (_config.get("warp_factor_min", 0.9), _config.get("warp_factor_max", 1.1)),
            _config.get("sampling_type", "random"),
            _config.get("boundary_freq", 4800))
