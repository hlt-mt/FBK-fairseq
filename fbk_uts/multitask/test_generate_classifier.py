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
import unittest
from argparse import Namespace
from unittest.mock import patch

import numpy as np

from examples.speech_to_text.models.multitask_conformer import MultitaskConformer, conformer_multitask_s
from examples.speech_to_text.scripts.generate_multitask_classifier import generate_probs, AccuracyScorer
from fbk_uts.multitask.test_speech_aux_classification_dataset import AuxiliaryClassificationDatasetSetup
from tests import utils as test_utils


class MultiTaskConformerTestCase(AuxiliaryClassificationDatasetSetup, unittest.TestCase):
    base_args = None

    @classmethod
    def setUpClass(cls):
        cls.base_args = Namespace()
        cls.base_args.encoder_embed_dim = 16
        cls.base_args.input_feat_per_channel = 4
        cls.base_args.input_channels = 1
        cls.base_args.max_source_positions = 10
        cls.base_args.max_target_positions = 20
        cls.base_args.encoder_layers = 2
        cls.base_args.ctc_compress_strategy = "none"
        cls.base_args.criterion = "label_smoothed_cross_entropy"
        cls.base_args.no_syncbatchnorm = True
        cls.base_args.ctc_compress_max_out_size = -1
        cls.base_args.ctc_compress_fixed_ratio = 4
        cls.base_args.reverted_classifier = False
        conformer_multitask_s(cls.base_args)

    @patch('fairseq.data.audio.speech_to_text_dataset.get_features_or_waveform')
    def test_basic(self, mock_get_features_or_waveform):
        mock_get_features_or_waveform.return_value = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        samples = self.ds.collater([self.ds[0], self.ds[1], self.ds[2]])
        task = test_utils.TestTranslationTask.setup_task(self.base_args, self.src_dict, self.tgt_dict)
        task.data_cfg = self.ds.data_cfg
        model = MultitaskConformer.build_model(self.base_args, task)
        probs = generate_probs([model], samples)
        self.assertListEqual(list(probs.shape), [3, 2])
        self.assertTrue(all(1 >= p[0] >= 0 for p in probs))

    def test_accuracy_scorer(self):
        scorer = AccuracyScorer()
        self.assertAlmostEqual(scorer.score(), 0.0)
        scorer.add(1, 0)
        self.assertAlmostEqual(scorer.score(), 0.0)
        scorer.add(0, 0)
        self.assertAlmostEqual(scorer.score(), 0.5)
        scorer.add(1, 1)
        self.assertAlmostEqual(scorer.score(), 2 / 3)
        self.assertEqual(
            scorer.result_string(), "Accuracy: 0.67. Class accuracy: {'0': 1.00, '1': 0.50}")


if __name__ == '__main__':
    unittest.main()
