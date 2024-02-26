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

import torch
from torch import Tensor

from examples.speech_to_text.occlusion_explanation.scorers import register_scorer, Scorer


@register_scorer("predicted_token_diff")
class PredictedTokenDifferenceScorer(Scorer):
    """
    Assign the attribution scores based on the difference between the original and
    perturbed probabilities of the token of the constrained decoding, namely the token
    selected in the beam search with the original generation
    """
    @staticmethod
    def get_prob_diff(
            orig_probs: Tensor,
            perturb_probs: Tensor,
            target: Tensor) -> Tensor:
        """
        Get the probability difference.
        """
        token_orig_probs = torch.take_along_dim(orig_probs, target.unsqueeze(-1), dim=2)
        token_perturb_probs = torch.take_along_dim(perturb_probs, target.unsqueeze(-1), dim=2)
        return token_orig_probs - token_perturb_probs