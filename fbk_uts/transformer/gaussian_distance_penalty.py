# Copyright 2021 FBK

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
from argparse import Namespace
from examples.speech_to_text.models.s2t_transformer_fbk import s2t_transformer_s, S2TTransformerModel
from examples.speech_to_text.modules.transformer_layer_penalty import TransformerEncoderLayerPenalty
from fairseq.data import Dictionary


class GaussianDistancePenaltyTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.base_args = Namespace()
        cls.base_args.encoder_embed_dim = 16
        cls.base_args.input_feat_per_channel = 5
        cls.base_args.input_channels = 1
        cls.base_args.max_source_positions = 10
        cls.base_args.encoder_layers = 8
        cls.base_args.criterion = "label_smoothed_crossentropy"
        cls.base_args.distance_penalty = "gauss"
        cls.base_args.init_variance = 5.0
        s2t_transformer_s(cls.base_args)
        cls.fake_dict = Dictionary()

    def test_gaussian_penalty(self):
        self.assertTrue(TransformerEncoderLayerPenalty(self.base_args))

    if __name__ == '__main__':
        unittest.main()
