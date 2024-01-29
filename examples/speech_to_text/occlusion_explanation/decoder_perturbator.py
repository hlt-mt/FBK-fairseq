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

import torch
from torch import Tensor

from examples.speech_to_text.occlusion_explanation import (
    register_perturbator,
    OcclusionDecoderEmbeddingsPerturbator)


@register_perturbator("continuous_embed")
class OcclusionDecoderEmbeddingsPerturbatorContinuous(
        OcclusionDecoderEmbeddingsPerturbator):
    """
    Class for implementing continuous occlusion perturbations.
    In this method, each value in the input data is zoroed out
    independently of the others.
    """
    def __call__(self, embeddings: Tensor) -> Tuple[Tensor, Tensor]:
        # embeddings Tensor has shape (B x T x C)
        masks = (torch.rand(embeddings.shape, device=embeddings.device) > self.p).to(embeddings.dtype)
        embeddings *= masks
        return embeddings, masks


@register_perturbator("discrete_embed")
class OcclusionDecoderEmbeddingsPerturbatorDiscrete(
        OcclusionDecoderEmbeddingsPerturbator):
    """
    Class for implementing discrete occlusion perturbations.
    In this method, entire token embeddings are zeroed out.
    """
    def __call__(self, embeddings: Tensor) -> Tuple[Tensor, Tensor]:
        # embeddings Tensor has shape (B x T x C)
        masks = (torch.rand(
            embeddings.shape[0], embeddings.shape[1], device=embeddings.device) > self.p).to(embeddings.dtype)
        embeddings *= masks.view(masks.shape[0], masks.shape[1], 1)
        return embeddings, masks
