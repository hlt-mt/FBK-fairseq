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
import torch

import tests.utils
from examples.speech_to_text.criterions.ctc_multi_loss import CTCMultiLoss
from examples.speech_to_text.data.speech_to_text_dataset_with_src import SpeechToTextDatasetWithSrc, S2TDataConfigSrc
from examples.speech_to_text.models.conformer import ConformerEncoder
from examples.speech_to_text.models.conformer_triangle import ConformerTriangle
from examples.speech_to_text.models.s2t_transformer_fbk_triangle import s2t_transformer_triangle_s,\
    S2TTransformerTriangle
from fairseq.data import Dictionary


class MockS2TDataConfigSrc(S2TDataConfigSrc):
    def __init__(self):
        self.config = {}


class TriangleTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.src_dict = Dictionary()
        src_lines = ["I like quokkas", "I like tortoises", "I like elephants"]
        for l in src_lines:
            self.src_dict.encode_line(l)
        self.tgt_dict = Dictionary()
        tgt_lines = ["Mi piacciono i quokka", "Mi piacciono le tartarughe", "Mi piacciono gli elefanti"]
        for l in tgt_lines:
            self.tgt_dict.encode_line(l)
        self.ds = SpeechToTextDatasetWithSrc(
            "quokka",
            True,
            MockS2TDataConfigSrc(),
            ["f1.wav", "f2.wav", "f3.wav"],
            [30, 100, 27],
            src_lines,
            tgt_lines,
            ["s1", "s2", "s3"],
            tgt_dict=self.tgt_dict,
            src_dict=self.src_dict,
        )
        args = Namespace()
        args.underlying_criterion = "cross_entropy_dualdecoder"
        args.criterion = "ctc_multi_loss"
        args.ctc_encoder_layer = 1
        args.ctc_encoder_layer = 1
        args.ctc_compress_strategy = "none"
        args.ctc_post_process = "letter"
        args.wer_args = None
        args.wer_kenlm_model = None
        args.zero_infinity = True
        args.sentence_avg = True
        args.label_smoothing = 0.1
        args.auxiliary_loss_weight = 0.5
        args.primary_loss_weight = 0.5
        args.ctc_weight = 0.5
        s2t_transformer_triangle_s(args)
        args.encoder_layers = 2
        args.decoder_layers = 2
        args.input_feat_per_channel = 4
        args.input_channels = 1
        args.max_source_positions = 10
        self.args = args

    @patch('fairseq.data.audio.speech_to_text_dataset.get_features_or_waveform')
    def test_ctc(self, mock_get_features_or_waveform):
        mock_get_features_or_waveform.return_value = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        task = tests.utils.TestTranslationTask.setup_task(
            self.args, self.src_dict, self.tgt_dict
        )
        samples = self.ds.collater([self.ds[0], self.ds[1], self.ds[2]])
        model = S2TTransformerTriangle.build_model(self.args, task)
        criterion = CTCMultiLoss(self.args, task)
        loss, _, logging_out = criterion.forward(model, samples)
        self.assertTrue(loss > 0)

    @patch('fairseq.data.audio.speech_to_text_dataset.get_features_or_waveform')
    def test_conformer(self, mock_get_features_or_waveform):
        mock_get_features_or_waveform.return_value = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        task = tests.utils.TestTranslationTask.setup_task(
            self.args, self.src_dict, self.tgt_dict
        )
        samples = self.ds.collater([self.ds[0], self.ds[1], self.ds[2]])
        model = ConformerTriangle.build_model(self.args, task)
        criterion = CTCMultiLoss(self.args, task)
        with patch.object(torch.nn.modules.SyncBatchNorm, "forward", side_effect=lambda x: x):
            loss, _, logging_out = criterion.forward(model, samples)
            self.assertTrue(loss > 0)
        self.assertIsInstance(model.encoder, ConformerEncoder)


if __name__ == '__main__':
    unittest.main()
