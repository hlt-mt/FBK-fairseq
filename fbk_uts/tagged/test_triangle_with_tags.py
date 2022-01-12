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
from unittest.mock import patch

import numpy as np

from examples.speech_to_text.models.s2t_transformer_triangle_with_tags import s2t_triangle_with_tags_s, \
    S2TTransformerTriangleWithTags
from fbk_uts.tagged.test_speech_tagged_dataset import TaggedDatasetSetup
from tests import utils as test_utils


class TriangleWithTagsTestCase(TaggedDatasetSetup, unittest.TestCase):
    @patch('fairseq.data.audio.speech_to_text_dataset.get_features_or_waveform')
    def test_basic(self, mock_get_features_or_waveform):
        mock_get_features_or_waveform.return_value = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        samples = self.ds.collater([self.ds[0], self.ds[1], self.ds[2]])
        args = Namespace()
        args.input_feat_per_channel = 4
        args.input_channels = 1
        args.max_source_positions = 10
        args.criterion = "cross_entropy"
        args.ctc_compress_strategy = "none"
        s2t_triangle_with_tags_s(args)
        d = test_utils.dummy_dictionary(20)
        task = test_utils.TestTranslationTask.setup_task(args, d, d)
        task.data_cfg = self.ds.data_cfg
        model = S2TTransformerTriangleWithTags.build_model(args, task)
        model_out = model(**samples["net_input"])
        self.assertIn("tags", model_out[0][1])
        self.assertIn("tags", model_out[1][1])


if __name__ == '__main__':
    unittest.main()
