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
import copy
import unittest
from argparse import Namespace
from unittest.mock import patch

import numpy as np
import torch
from examples.speech_to_text.models.speechformer import SpeechformerEncoder

import tests.utils
from examples.speech_to_text.criterions.ctc_multi_loss import CTCMultiLoss
from examples.speech_to_text.data.speech_to_text_dataset_with_src import SpeechToTextDatasetWithSrc, S2TDataConfigSrc
from examples.speech_to_text.models.conformer import ConformerEncoder
from examples.speech_to_text.models.conformer_triangle import ConformerTriangle
from examples.speech_to_text.models.s2t_transformer_fbk_triangle import s2t_transformer_triangle_s,\
    S2TTransformerTriangle
from examples.speech_to_text.models.speechformer_triangle import SpeechformerTriangle
from fairseq import utils
from fairseq.logging.meters import MetersDict
from fbk_uts.base_utilities import BaseSpeechTestCase


class TriangleTestCase(unittest.TestCase, BaseSpeechTestCase):
    @patch('fairseq.data.audio.speech_to_text_dataset.get_features_or_waveform')
    def setUp(self, mock_get_features_or_waveform) -> None:
        mock_get_features_or_waveform.return_value = np.random.random((200, 4))
        self.init_sample_dataset(SpeechToTextDatasetWithSrc)
        args = Namespace()
        args.underlying_criterion = "cross_entropy_dualdecoder"
        args.criterion = "ctc_multi_loss"
        args.ctc_encoder_layer = 1
        args.ctc_encoder_layer = 1
        args.ctc_compress_strategy = "none"
        args.ctc_post_process = "letter"
        args.wer_args = None
        args.wer_kenlm_model = None
        args.zero_infinity = False
        args.sentence_avg = True
        args.label_smoothing = 0.1
        args.auxiliary_loss_weight = 0.8
        args.primary_loss_weight = 0.2
        args.ctc_weight = 0.5
        s2t_transformer_triangle_s(args)
        args.encoder_layers = 2
        args.decoder_layers = 2
        args.input_feat_per_channel = 4
        args.input_channels = 1
        args.max_source_positions = 10
        self.args = args
        self.task = tests.utils.TestTranslationTask.setup_task(
            self.args, self.src_dict, self.tgt_dict
        )
        self.samples = self.ds.collater([self.ds[0], self.ds[1], self.ds[2]])

    def test_ctc(self):
        model = S2TTransformerTriangle.build_model(self.args, self.task)
        criterion = CTCMultiLoss(self.args, self.task)
        loss, _, _ = criterion.forward(model, self.samples)
        self.assertTrue(loss > 0)

    @patch("fairseq.logging.metrics.get_active_aggregators")
    def test_ctc_logging_out(self, mock_get_active_aggregators):
        metrics_logged = MetersDict()
        mock_get_active_aggregators.return_value = [metrics_logged]
        model = S2TTransformerTriangle.build_model(self.args, self.task)
        criterion = CTCMultiLoss(self.args, self.task)
        _, _, logging_out = criterion.forward(model, self.samples)
        self.assertIn("auxiliary_loss", logging_out)
        self.assertIn("primary_loss", logging_out)
        self.assertIn("real_loss", logging_out)
        self.assertIn("ctc_loss", logging_out)
        self.assertAlmostEqual(
            logging_out["real_loss"],
            self.args.primary_loss_weight * logging_out["primary_loss"] +
            self.args.auxiliary_loss_weight * logging_out["auxiliary_loss"],
            places=4)
        self.assertAlmostEqual(
            logging_out["loss"],
            logging_out["real_loss"] + self.args.ctc_weight * logging_out["ctc_loss"],
            places=4)
        CTCMultiLoss.reduce_metrics([logging_out])
        self.assertIn("auxiliary_loss", metrics_logged)
        self.assertIn("primary_loss", metrics_logged)
        self.assertIn("real_loss", metrics_logged)
        self.assertIn("ctc_loss", metrics_logged)
        self.assertAlmostEqual(
            metrics_logged["real_loss"].avg,
            self.args.primary_loss_weight * utils.item(metrics_logged["primary_loss"].avg) +
            self.args.auxiliary_loss_weight * utils.item(metrics_logged["auxiliary_loss"].avg),
            places=4)
        self.assertAlmostEqual(
            metrics_logged["loss"].avg,
            metrics_logged["real_loss"].avg +
            self.args.ctc_weight * metrics_logged["ctc_loss"].avg,
            places=4)

    def test_conformer(self):
        model = ConformerTriangle.build_model(self.args, self.task)
        criterion = CTCMultiLoss(self.args, self.task)
        with patch.object(torch.nn.modules.SyncBatchNorm, "forward", side_effect=lambda x: x):
            loss, _, logging_out = criterion.forward(model, self.samples)
            self.assertTrue(loss > 0)
        self.assertIsInstance(model.encoder, ConformerEncoder)

    def test_speechformer(self):
        speechformer_args = copy.deepcopy(self.args)
        speechformer_args.transformer_after_compression = True
        model = SpeechformerTriangle.build_model(speechformer_args, self.task)
        criterion = CTCMultiLoss(speechformer_args, self.task)
        loss, _, logging_out = criterion.forward(model, self.samples)
        self.assertTrue(loss > 0)
        self.assertIsInstance(model.encoder, SpeechformerEncoder)


if __name__ == '__main__':
    unittest.main()
