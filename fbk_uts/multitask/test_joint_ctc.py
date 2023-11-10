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
import copy
import unittest
from argparse import Namespace
from unittest.mock import patch

import numpy as np
import torch

from examples.speech_to_text.criterions.cross_entropy_auxiliary_ctc import JointCrossEntropyCtcLoss
from examples.speech_to_text.criterions.ctc_multi_loss import CTCMultiLoss
from examples.speech_to_text.data.speech_to_text_dataset_with_src import SpeechToTextDatasetWithSrc
from examples.speech_to_text.models.conformer_joint_ctc import JointCtcConformer, conformer_joint_ctc_s
from fbk_uts.base_utilities import BaseSpeechTestCase
from tests import utils as test_utils


class JointCtcConformerTestCase(BaseSpeechTestCase, unittest.TestCase):
    base_args = None

    @classmethod
    def setUpClass(cls):
        cls.base_args = Namespace()
        cls.base_args.encoder_embed_dim = 16
        cls.base_args.input_feat_per_channel = 4
        cls.base_args.input_channels = 1
        cls.base_args.max_source_positions = 10
        cls.base_args.max_target_positions = 20
        cls.base_args.encoder_layers = 8
        cls.base_args.ctc_compress_strategy = "none"
        cls.base_args.criterion = "joint_cross_entropy_ctc"
        cls.base_args.no_syncbatchnorm = True
        cls.base_args.ctc_compress_max_out_size = -1
        cls.base_args.ctc_compress_fixed_ratio = 4
        cls.base_args.primary_loss_weight = 0.5
        cls.base_args.auxiliary_loss_weight = 0.5
        cls.base_args.label_smoothing = 0.1
        cls.base_args.ignore_prefix_size = 0
        cls.base_args.sentence_avg = True
        conformer_joint_ctc_s(cls.base_args)

    @patch('fairseq.data.audio.speech_to_text_dataset.get_features_or_waveform')
    def test_basic(self, mock_get_features_or_waveform):
        mock_get_features_or_waveform.return_value = np.array(
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        self.init_sample_dataset(SpeechToTextDatasetWithSrc)
        samples = self.ds.collater([self.ds[0], self.ds[1], self.ds[2]])
        task = test_utils.TestTranslationTask.setup_task(self.base_args, self.src_dict, self.tgt_dict)
        task.data_cfg = self.ds.data_cfg
        model = JointCtcConformer.build_model(self.base_args, task)
        model_out = model(**samples["net_input"])
        self.assertEqual(len(model_out), 2)
        self.assertListEqual(list(model_out[1][0].shape), [1, 3, len(self.tgt_dict)])
        self.assertIn("<ctc_blank>", self.tgt_dict)
        self.assertTrue(model.get_normalized_probs(model_out[0], True).batch_first)

    @patch('fairseq.data.audio.speech_to_text_dataset.get_features_or_waveform')
    def test_ctc(self, mock_get_features_or_waveform):
        args = copy.deepcopy(self.base_args)
        args.ctc_encoder_layer = 4
        args.ctc_compress_strategy = "avg"
        args.criterion = "ctc_multi_loss"
        mock_get_features_or_waveform.return_value = np.array(
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        self.init_sample_dataset(SpeechToTextDatasetWithSrc)
        samples = self.ds.collater([self.ds[0], self.ds[1], self.ds[2]])
        task = test_utils.TestTranslationTask.setup_task(args, self.src_dict, self.tgt_dict)
        task.data_cfg = self.ds.data_cfg
        model = JointCtcConformer.build_model(args, task)
        model_out = model(**samples["net_input"])
        self.assertEqual(len(model_out), 2)
        self.assertEqual(len(model_out[0]), 2)
        self.assertEqual(len(model_out[0][1]), 2)
        self.assertListEqual(list(model_out[0][1][0].shape), [1, 3, len(self.tgt_dict)])

    def test_padding(self):
        args = copy.deepcopy(self.base_args)
        args.ctc_encoder_layer = 4
        args.ctc_compress_strategy = "avg"
        args.criterion = "ctc_multi_loss"
        self.init_sample_dataset(SpeechToTextDatasetWithSrc)
        task = test_utils.TestTranslationTask.setup_task(args, self.src_dict, self.tgt_dict)
        task.data_cfg = self.ds.data_cfg
        fake_sample = torch.rand(2, 27, 4)
        fake_sample[1, 13:, :] = 0
        fake_lengths = torch.LongTensor([27, 13])
        model = JointCtcConformer.build_model(self.base_args, task)
        model.eval()
        net_out = model(
            fake_sample, fake_lengths,  prev_output_tokens=torch.tensor([[0], [0]]))
        ctc_out = net_out[1][0]
        net_out_no_pad = model(
            fake_sample[1:, :13, :], fake_lengths[1:], prev_output_tokens=torch.tensor([[0]]))
        ctc_out_no_pad = net_out_no_pad[1][0]
        torch.testing.assert_allclose(ctc_out[:4, 1, :], ctc_out_no_pad[:, 0, :])

    @patch('fairseq.data.audio.speech_to_text_dataset.get_features_or_waveform')
    def test_ctc_joint_loss(self, mock_get_features_or_waveform):
        mock_get_features_or_waveform.return_value = np.random.random((27, 4))
        args = copy.deepcopy(self.base_args)
        self.init_sample_dataset(SpeechToTextDatasetWithSrc)
        task = test_utils.TestTranslationTask.setup_task(args, self.src_dict, self.tgt_dict)
        task.data_cfg = self.ds.data_cfg
        samples = self.ds.collater([self.ds[0], self.ds[1], self.ds[2]])
        loss = JointCrossEntropyCtcLoss(task, True, 0.1)
        model = JointCtcConformer.build_model(self.base_args, task)
        model.eval()
        loss_out, _, _ = loss(model, samples)
        self.assertGreater(loss_out, 0.)

    @patch('fairseq.data.audio.speech_to_text_dataset.get_features_or_waveform')
    def test_ctc_multiloss(self, mock_get_features_or_waveform):
        mock_get_features_or_waveform.return_value = np.random.random((27, 4))
        args = copy.deepcopy(self.base_args)
        args.criterion = "ctc_multi_loss"
        args.ctc_weight = 0.5
        args.ctc_post_process = None
        args.wer_args = None
        args.wer_kenlm_model = None
        args.zero_infinity = True
        args.ctc_encoder_layer = 2
        args.underlying_criterion = "joint_cross_entropy_ctc"
        self.init_sample_dataset(SpeechToTextDatasetWithSrc)
        task = test_utils.TestTranslationTask.setup_task(args, self.src_dict, self.tgt_dict)
        task.data_cfg = self.ds.data_cfg
        samples = self.ds.collater([self.ds[0], self.ds[1], self.ds[2]])
        loss = CTCMultiLoss(args, task)
        model = JointCtcConformer.build_model(args, task)
        model.eval()
        loss_out, _, _ = loss(model, samples)
        self.assertGreater(loss_out, 0.)

    @patch('fairseq.data.audio.speech_to_text_dataset.get_features_or_waveform')
    def test_multilingual(self, mock_get_features_or_waveform):
        mock_get_features_or_waveform.return_value = np.random.random((27, 4))
        args = copy.deepcopy(self.base_args)
        args.ignore_prefix_size = 1
        self.init_sample_dataset(SpeechToTextDatasetWithSrc)
        task = test_utils.TestTranslationTask.setup_task(args, self.src_dict, self.tgt_dict)
        self.ds.data_cfg.config["prepend_tgt_lang_tag"] = True
        self.ds.tgt_langs = ["<lang:it>", "<lang:aa>", "<lang:it>"]
        self.tgt_dict.add_symbol("<lang:it>")
        self.tgt_dict.add_symbol("<lang:aa>")
        task.data_cfg = self.ds.data_cfg
        samples = self.ds.collater([self.ds[0], self.ds[1], self.ds[2]])
        loss = JointCrossEntropyCtcLoss(task, True, 0.1)
        model = JointCtcConformer.build_model(args, task)
        model.eval()
        loss_out, _, _ = loss(model, samples)
        self.assertGreater(loss_out, 0.)


if __name__ == '__main__':
    unittest.main()
