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

import importlib
import os
import logging
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Optional, List

from torch import Tensor


LOGGER = logging.getLogger(__name__)


class OcclusionFbankPerturbator(ABC):
    """
    Base class to perform occlusion perturbations of filterbanks.
    """
    @classmethod
    @abstractmethod
    def from_config_dict(cls, config: Dict = None):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        pass


class OcclusionDecoderEmbeddingsPerturbator(ABC):
    """
    Base class to perform occlusion perturbations of target embeddings.
    """
    def __init__(
            self, no_position_occlusion: bool = False, p: float = 0.5):
        self.no_position_occlusion = no_position_occlusion
        self.p = p
        LOGGER.info(f"Masking decoder embeddings with probability of {self.p}.")

    @classmethod
    def from_config_dict(cls, config: Dict = None):
        if config is None:
            return cls(no_position_occlusion=True, p=0.5)
        else:
            _config = config
            decoder_occlusion = _config.get("decoder_occlusion", {})
            no_position_occlusion = decoder_occlusion.get("no_position_occlusion", True)
            p = decoder_occlusion.get("p", 0.5)
            assert 0. <= p < 1.
            return cls(no_position_occlusion, p)

    def __call__(
            self,
            embeddings: Tensor,
            occlusion_masks: Optional[Tensor] = None,
            last_tokens_to_perturb: Optional[List[int]] = None,
            force_masks_length: Optional[bool] = False) -> Tuple[Tensor, Tensor]:
        """
        Apply occlusion to the embeddings.
        If the occlusion_masks are not provided, generate them.
        The occlusion_masks can be provided e.g. when explaining gender because the same mask
        should be used for the original and swapped hypotheses.

        Args:
            embeddings (Tensor): The embeddings to occlude (B x T x C).
            occlusion_masks (Tensor): The occlusion masks (B x T).
            last_tokens_to_perturb (List[int]): Tokens after this position will not be occluded.
                This is useful when explaining only specific parts of the hypothesis, e.g. for gender.
            force_masks_length (bool): If True, the occlusion masks will be adjusted (truncated or padded)
                to match the embeddings, this is useful e.g. when studying gender.

        Returns:
            Tuple[Tensor, Tensor]: The occluded embeddings and the occlusion masks.        
        """
        if occlusion_masks is None:
            occlusion_masks = self._generate_occlusion_masks(embeddings)
        if last_tokens_to_perturb:
            for i, last_token in enumerate(last_tokens_to_perturb):
                # tokens after this position will not be occluded
                occlusion_masks[i, last_token + 1:] = 1.  # +1 because of the bos token
        if force_masks_length:
            occlusion_masks = self._adjust_occlusion_masks_length(occlusion_masks, embeddings)
        embeddings = self._apply_occlusion(embeddings, occlusion_masks)
        return embeddings, occlusion_masks


PERTURBATION_REGISTRY = {}
PERTURBATION_CLASS_NAMES = set()


def register_perturbator(name):
    def register_perturbation_cls(cls):
        if name in PERTURBATION_REGISTRY:
            raise ValueError(
                f"Cannot register duplicate perturbation ({name})")
        if not issubclass(cls, (OcclusionFbankPerturbator, OcclusionDecoderEmbeddingsPerturbator)):
            raise ValueError(
                f"Perturbation ({name}: {cls.__name__}) must extend "
                "OcclusionFbankPerturbator or OcclusionDecoderEmbeddingsPerturbator")
        if cls.__name__ in PERTURBATION_CLASS_NAMES:
            raise ValueError(
                f"Cannot register perturbation with duplicate class name ({cls.__name__})")
        PERTURBATION_REGISTRY[name] = cls
        PERTURBATION_CLASS_NAMES.add(cls.__name__)
        LOGGER.debug(f"Occlusion perturbation registered: {name}.")
        return cls
    return register_perturbation_cls


def get_perturbator(name):
    return PERTURBATION_REGISTRY[name]


for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module_name = file[:file.find('.py')]
        importlib.import_module(
            'examples.speech_to_text.occlusion_explanation.perturbators.' + module_name)
