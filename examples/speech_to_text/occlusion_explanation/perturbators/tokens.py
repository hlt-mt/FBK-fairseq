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

from typing import List, Optional, Tuple

import torch
from torch import Tensor

from examples.speech_to_text.occlusion_explanation.perturbators import (
    register_perturbator,
    OcclusionDecoderEmbeddingsPerturbator)


@register_perturbator("continuous_embed")
class OcclusionDecoderEmbeddingsPerturbatorContinuous(
        OcclusionDecoderEmbeddingsPerturbator):
    """
    Class for implementing continuous occlusion perturbations.
    In this method, each value in the input data is zeroed out
    independently of the others.
    """
    @staticmethod
    def _adjust_occlusion_masks_length(occlusion_masks, embeddings):
        """
        The occlusion masks will be adjusted to match the embeddings. 
        If the occlusion_masks are longer than the embeddings, they will be truncated.
        If they are shorter, they are padded with ones (this can happen when studying gender terms,
        if the swapped hypothesis is longer or shorter than the original one).

        Args:
            occlusion_masks (torch.Tensor): The initial occlusion masks.
            embeddings (torch.Tensor): The embeddings tensor which is going to be perturbed.

        Returns:
            torch.Tensor: The adjusted occlusion masks.
        """
        if embeddings.shape[1] > occlusion_masks.shape[1]:
            occlusion_masks = torch.cat(
                [occlusion_masks, torch.ones(
                    embeddings.shape[0], embeddings.shape[1] - occlusion_masks.shape[1],
                    embeddings.shape[2], device=embeddings.device, dtype=occlusion_masks.dtype)], dim=1)
        elif embeddings.shape[1] < occlusion_masks.shape[1]:
            occlusion_masks = occlusion_masks[:, :embeddings.shape[1], :]
        return occlusion_masks
    
    def _generate_occlusion_masks(self, embeddings: Tensor) -> Tensor:
        """
        Args:
            embeddings (torch.Tensor): The embeddings tensor which is going to be perturbed.

        Returns:
            torch.Tensor: The occlusion masks.
        """
        random_values = torch.rand(embeddings.shape, device=embeddings.device)
        return random_values.ge(self.p).to(embeddings.dtype)

    def _apply_occlusion(self, embeddings: Tensor, occlusion_masks: Tensor) -> Tensor:
        """
        Args:
            embeddings (torch.Tensor): The embeddings tensor which is going to be perturbed.
            occlusion_masks (torch.Tensor): The occlusion masks.

        Returns:
            torch.Tensor: The perturbed embeddings.
        """
        return embeddings * occlusion_masks


@register_perturbator("discrete_embed")
class OcclusionDecoderEmbeddingsPerturbatorDiscrete(
        OcclusionDecoderEmbeddingsPerturbator):
    """
    Class for implementing discrete occlusion perturbations.
    In this method, entire token embeddings are zeroed out.
    """
    @staticmethod
    def _adjust_occlusion_masks_length(occlusion_masks, embeddings):
        """
        The occlusion masks will be adjusted to match the embeddings. 
        If the occlusion_masks are longer than the embeddings, they will be truncated.
        If they are shorter, they are padded with ones (this can happen when studying gender terms,
        if the swapped hypothesis is longer or shorter than the original one).

        Args:
            occlusion_masks (torch.Tensor): The initial occlusion masks.
            embeddings (torch.Tensor): The embeddings tensor which is going to be perturbed.

        Returns:
            torch.Tensor: The adjusted occlusion masks.
        """
        if embeddings.shape[1] > occlusion_masks.shape[1]:
            occlusion_masks = torch.cat(
                [occlusion_masks, torch.ones(
                    embeddings.shape[0], embeddings.shape[1] - occlusion_masks.shape[1],
                    device=embeddings.device, dtype=occlusion_masks.dtype)], dim=1)
        elif embeddings.shape[1] < occlusion_masks.shape[1]:
            occlusion_masks = occlusion_masks[:, :embeddings.shape[1]]

        return occlusion_masks

    def _generate_occlusion_masks(self, embeddings: Tensor) -> Tensor:
        """
        Args:
            embeddings (torch.Tensor): The embeddings tensor which is going to be perturbed.

        Returns:
            torch.Tensor: The occlusion masks.
        """
        random_values = torch.rand(
            embeddings.shape[0], embeddings.shape[1], device=embeddings.device)
        return random_values.ge(self.p).to(embeddings.dtype)

    def _apply_occlusion(self, embeddings: Tensor, occlusion_masks: Tensor) -> Tensor:
        """
        Args:
            embeddings (torch.Tensor): The embeddings tensor which is going to be perturbed.
            occlusion_masks (torch.Tensor): The occlusion masks.

        Returns:
            torch.Tensor: The perturbed embeddings.
        """
        return embeddings * occlusion_masks.view(occlusion_masks.shape[0], occlusion_masks.shape[1], 1)
