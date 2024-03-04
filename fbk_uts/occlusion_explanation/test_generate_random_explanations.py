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

import unittest

import torch

from examples.speech_to_text.scripts.generate_random_explanations import generate_random_explanations


class TestRandomExplanations(unittest.TestCase):
    def setUp(self):
        self.index = 0
        self.fbank_heatmaps = torch.randn(456, 80)
        self.tgt_embed_heatmap = torch.randn(14)

    # check size
    def test_generate_random_explanations(self):
        random_explanations = generate_random_explanations(
            (self.index, self.fbank_heatmaps, self.tgt_embed_heatmap), 1)
        self.assertTrue(random_explanations["fbank_heatmap"].size(), (14, 456, 80))
        self.assertTrue(random_explanations["tgt_embed_heatmap"].size(), (14, 14, 1))
        random_explanations = generate_random_explanations(
            (self.index, self.fbank_heatmaps, self.tgt_embed_heatmap), 1024, 1)
        self.assertTrue(random_explanations["fbank_heatmap"].size(), (14, 456, 1))
        self.assertTrue(random_explanations["tgt_embed_heatmap"].size(), (14, 14, 1024))

    # test randomness of explanations that must be different from original data
    def test_randomness_explanations(self):
        random_explanations = generate_random_explanations(
            (self.index, self.fbank_heatmaps, self.tgt_embed_heatmap), 1)
        # loop over the number of tokens, in this case 14
        for i in range(14):
            self.assertFalse(
                torch.equal(random_explanations["fbank_heatmap"][i], self.fbank_heatmaps))
            self.assertFalse(
                torch.equal(random_explanations["tgt_embed_heatmap"][i], self.tgt_embed_heatmap))

    def test_seed_random_generation(self):
        fbank_explanations_prev = torch.tensor([])
        tgt_explanations_prev = torch.tensor([])
        # check that different seeds generate different random explanations
        for i in range(3):
            random_explanations = generate_random_explanations(
                (self.index, self.fbank_heatmaps, self.tgt_embed_heatmap), 1, None, i)
            self.assertFalse(torch.equal(fbank_explanations_prev, random_explanations["fbank_heatmap"]))
            self.assertFalse(torch.equal(tgt_explanations_prev, random_explanations["tgt_embed_heatmap"]))
            fbank_explanations_prev = random_explanations["fbank_heatmap"]
            tgt_explanations_prev = random_explanations["tgt_embed_heatmap"]
        # check that seed works by comparing two generation with same seeds (i.e., 2)
        random_explanations = generate_random_explanations(
            (self.index, self.fbank_heatmaps, self.tgt_embed_heatmap), 1, None, 2)
        self.assertTrue(torch.equal(fbank_explanations_prev, random_explanations["fbank_heatmap"]))
        self.assertTrue(torch.equal(tgt_explanations_prev, random_explanations["tgt_embed_heatmap"]))


if __name__ == '__main__':
    unittest.main()
