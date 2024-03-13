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

import os

import unittest
from torch import Tensor

from examples.speech_to_text.occlusion_explanation.utils import read_feature_attribution_maps_from_h5


class TestSentenceLevelExplanation(unittest.TestCase):
    def setUp(self) -> None:
        current_directory = os.path.dirname(__file__)
        self.explanation_path = os.path.join(
            current_directory, '../xai_metrics/mock_data/explanations.h5')

    def test_read_feature_attribution_maps_from_h5(self):
        explanations = read_feature_attribution_maps_from_h5(self.explanation_path)
        self.assertTrue(1 in explanations)
        self.assertTrue(2 in explanations)
        self.assertTrue("fbank_heatmap" in explanations[1])
        self.assertTrue("fbank_heatmap" in explanations[2])
        self.assertTrue("tgt_embed_heatmap" in explanations[1])
        self.assertTrue("tgt_embed_heatmap" in explanations[2])
        self.assertTrue(isinstance(explanations[1]["fbank_heatmap"], Tensor))
        self.assertTrue(isinstance(explanations[1]["tgt_embed_heatmap"], Tensor))
        self.assertTrue(isinstance(explanations[2]["fbank_heatmap"], Tensor))
        self.assertTrue(isinstance(explanations[2]["tgt_embed_heatmap"], Tensor))


if __name__ == '__main__':
    unittest.main()
