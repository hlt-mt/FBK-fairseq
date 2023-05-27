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
import csv
import logging
import random
import warnings
from typing import Tuple, Set

import numpy as np

import parselmouth

from fairseq.data.audio.feature_transforms import AudioFeatureTransform, register_audio_feature_transform


logger = logging.getLogger(__name__)


class PitchScalingBase(AudioFeatureTransform):
    """
    Base class to perform pitch (F0 and formants) scaling.
    """
    def __init__(self, gender_tsv: str, sampling_rate: int):
        assert gender_tsv is not None, "The path of MuST-Speaker file must be provided in the config file"
        with open(gender_tsv, 'r') as speakers_f:
            speakers_reader = csv.DictReader(
                speakers_f,
                delimiter="\t",
                quotechar=None,
                doublequote=False,
                lineterminator="\n",
                quoting=csv.QUOTE_NONE,)
            self.id_to_gender = {line["TALK-ID"]: line["TED-PRONOUN"].strip() for line in speakers_reader}
        self.sampling_rate = sampling_rate

    @property
    def extra_args(self) -> Set:
        return {"ids"}

    @property
    def duration_factor(self):
        return 1  # no lengthening of the sound

    def _not_applicable(self, gender_pronoun, ids):
        """
        Decides whether to apply or not the transform.
        """
        raise NotImplementedError

    def get_target_gender(self, from_male) -> bool:
        """
        According to the type of policy (and eventually the gender of the speaker) returns
        the gender of the speaker to simulate after the transformation (True being masculine
        and False feminine).
        """
        raise NotImplementedError

    def get_factor(self, from_male: bool, to_male: bool) -> float:
        """
        According to the gender of the speaker and the target gender, returns the scaling
        factor that will be used for formant_shift_ratio and pitch_range_factor.
        """
        raise NotImplementedError

    @staticmethod
    def _f0_parameters(from_male: bool) -> Tuple[int, int]:
        """
        According to the gender of the speaker, returns:
        - pitch_floor
        - pitch_ceiling
        """
        if from_male:
            return 75, 250
        else:
            return 100, 400

    @staticmethod
    def get_new_pitch(to_male: bool) -> float:
        """
        Returns new median pitch randomly sampled from a normal distribution.
        """
        if to_male:
            median_f0, sigma = 140, 20  # range 80-200 Hz for males
        else:
            median_f0, sigma = 250, 17  # range 200-300 Hz for females
        new_median_f0 = np.random.normal(median_f0, sigma)
        return new_median_f0

    def _pitch_estimation(
            self, sound_object: parselmouth.Sound,
            from_male: bool,
            ids: str = None) -> Tuple[np.array, np.array]:
        """
        Returns the F0 estimation of the original audio.
        """
        pitch_floor, pitch_ceiling = self._f0_parameters(from_male)
        f0 = sound_object.to_pitch_ac(
            pitch_floor=pitch_floor,
            pitch_ceiling=pitch_ceiling,
            time_step=0.8 / pitch_floor)
        f0_np = f0.selected_array['frequency']
        return f0, np.median(f0_np[f0_np != 0]).item()

    def _manipulate(
            self,
            sound_object: parselmouth.Sound,
            from_male: bool,
            to_male: bool,
            ids: str = None) -> object:
        """
        Manipulates the waveform.
        """
        try:
            f0, f0_med = self._pitch_estimation(sound_object, from_male)
        except parselmouth.PraatError:
            logger.debug(f"Pitch estimation failed for {ids}")
            return None
        if np.isnan(f0_med):
            logger.debug(f"During pitch estimation no voiced segments found in {ids}")
            return None
        new_f0 = self.get_new_pitch(to_male)
        scaling_factor = self.get_factor(from_male, to_male)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", category=parselmouth.PraatWarning)  # Set the filter for PraatWarning
            new_sound = parselmouth.praat.call(
                [sound_object, f0],
                "Change gender",
                scaling_factor,  # formant_shift_ratio
                new_f0,
                scaling_factor,  # pitch_range_factor
                self.duration_factor)
        if any(isinstance(warn.message, parselmouth.PraatWarning) for warn in w):
            logger.debug(f"During pitch manipulation no voiced segments found in {ids}")
            new_sound = None
        return new_sound

    def __call__(self, waveform: np.array, ids: str = None) -> np.array:
        gender_pronoun = self.id_to_gender[ids.split("_")[1]]
        if self._not_applicable(gender_pronoun, ids):
            return waveform
        from_male = gender_pronoun == "He"
        to_male = self.get_target_gender(from_male)
        sound_object = parselmouth.Sound(waveform, sampling_frequency=self.sampling_rate, start_time=0.0)
        new_sound = self._manipulate(sound_object, from_male, to_male, ids)
        if new_sound is not None:
            logger.debug(f"Pitch manipulation performed for {ids}")
            return new_sound.values
        else:
            return waveform


@register_audio_feature_transform("random_pitch")
class PitchScalingRandom(PitchScalingBase):
    """
    Class to perform pitch scaling from the original gender of the speaker
    to a random gender (either Male or Female). In case of scaling between
    the same gender, only the median F0 will be changed.
    """
    def __init__(self, gender_tsv: str, sampling_rate: int, p: float):
        super().__init__(gender_tsv, sampling_rate)
        self.p = p

    @classmethod
    def from_config_dict(cls, config=None):
        _config = {} if config is None else config
        return PitchScalingRandom(
            _config.get("gender_tsv", None),
            _config.get("sampling_rate", 16000),
            _config.get("p", 0.5))

    def _not_applicable(self, gender_pronoun, ids) -> bool:
        if gender_pronoun not in ["He", "She"]:
            logger.debug(f"Pitch manipulation not performed for {ids} because non-binary")
            return True
        elif random.random() > self.p:
            return True
        else:
            return False

    def get_target_gender(self, *kwargs) -> bool:
        if random.random() < 0.5:
            return True
        else:
            return False

    def get_factor(self, from_male: bool, to_male: bool) -> float:
        if (from_male and to_male) or (not from_male and not to_male):
            return 1.0
        else:
            if from_male:
                return 1.2
            else:
                return 0.8


@register_audio_feature_transform("opposite_pitch")
class PitchScalingOpposite(PitchScalingBase):
    """
    Class to perform pitch scaling from Male to Female or from Female to Male
    based on the original gender of the speaker.
    """
    def __init__(self, gender_tsv: str, sampling_rate: int, p_male: float, p_female: float):
        super().__init__(gender_tsv, sampling_rate)
        self.p_male = p_male
        self.p_female = p_female

    @classmethod
    def from_config_dict(cls, config=None):
        _config = {} if config is None else config
        return PitchScalingOpposite(
            _config.get("gender_tsv", None),
            _config.get("sampling_rate", 16000),
            _config.get("p_male", 0.7),
            _config.get("p_female", 0.3))

    def _not_applicable(self, gender_pronoun, ids) -> bool:
        if gender_pronoun not in ["He", "She"]:
            logger.debug(f"Pitch manipulation not performed for {ids} because non-binary")
            return True
        elif (gender_pronoun == "He" and random.random() > self.p_male) or \
                (gender_pronoun == "She" and random.random() > self.p_female):
            return True
        else:
            return False

    def get_target_gender(self, from_male: bool) -> bool:
        if from_male:
            return False
        else:
            return True

    def get_factor(self, from_male: bool, *kwargs) -> float:
        if from_male:
            return 1.2
        else:
            return 0.8
