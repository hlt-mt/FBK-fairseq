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

from examples.speech_to_text.models.multitask_conformer import MultitaskConformer, conformer_multitask_s
from fbk_uts.multitask.test_speech_aux_classification_dataset import AuxiliaryClassificationDatasetSetup
from tests import utils as test_utils


def find_gradient_fn_recursive(grad_fn, name):
    if grad_fn is None:
        return False
    if type(grad_fn).__name__ == name:
        return True
    for f in grad_fn.next_functions:
        if find_gradient_fn_recursive(f[0], name):
            return True
    return False


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
        cls.base_args.encoder_layers = 8
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
        model_out = model(**samples["net_input"])
        assert len(model_out) == 2
        assert list(model_out[1].shape) == [3, 2]

    @patch('fairseq.data.audio.speech_to_text_dataset.get_features_or_waveform')
    def test_ctc(self, mock_get_features_or_waveform):
        args = copy.deepcopy(self.base_args)
        args.ctc_encoder_layer = 4
        args.ctc_compress_strategy = "avg"
        args.criterion = "ctc_multi_loss"
        mock_get_features_or_waveform.return_value = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        samples = self.ds.collater([self.ds[0], self.ds[1], self.ds[2]])
        task = test_utils.TestTranslationTask.setup_task(args, self.src_dict, self.tgt_dict)
        task.data_cfg = self.ds.data_cfg
        model = MultitaskConformer.build_model(args, task)
        model_out = model(**samples["net_input"])
        assert len(model_out) == 2
        assert list(model_out[0][1].shape) == [3, 2]

    @patch('fairseq.data.audio.speech_to_text_dataset.get_features_or_waveform')
    def test_gradient_reversal(self, mock_get_features_or_waveform):
        args = copy.deepcopy(self.base_args)
        args.ctc_encoder_layer = 4
        args.reverted_classifier = True
        args.ctc_compress_strategy = "avg"
        args.criterion = "ctc_multi_loss"
        mock_get_features_or_waveform.return_value = np.array(
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        samples = self.ds.collater([self.ds[0], self.ds[1], self.ds[2]])
        task = test_utils.TestTranslationTask.setup_task(args, self.src_dict, self.tgt_dict)
        task.data_cfg = self.ds.data_cfg
        model = MultitaskConformer.build_model(args, task)
        model_out = model(**samples["net_input"])
        assert len(model_out) == 2
        assert list(model_out[0][1].shape) == [3, 2]
        assert find_gradient_fn_recursive(
            model_out[0][1].grad_fn, "GradientReversalFunctionBackward")

    def test_padding(self):
        args = copy.deepcopy(self.base_args)
        args.ctc_encoder_layer = 4
        args.ctc_compress_strategy = "avg"
        args.criterion = "ctc_multi_loss"
        task = test_utils.TestTranslationTask.setup_task(args, self.src_dict, self.tgt_dict)
        task.data_cfg = self.ds.data_cfg
        fake_sample = torch.rand(2, 27, 4)
        fake_sample[1, 13:, :] = 0
        fake_lengths = torch.LongTensor([27, 13])
        model = MultitaskConformer.build_model(self.base_args, task)
        model.eval()
        net_out = model(
            fake_sample, fake_lengths,  prev_output_tokens=torch.tensor([[0], [0]]))
        classification_out = net_out[1][1]
        net_out_no_pad = model(
            fake_sample[1:, :13, :], fake_lengths[1:], prev_output_tokens=torch.tensor([[0]]))
        classification_out_no_pad = net_out_no_pad[1][0]
        for i in [0, 1]:
            self.assertAlmostEqual(
                classification_out[i].item(), classification_out_no_pad[i].item(), places=5)

    def test_gradient_reversal_freezing(self):
        args = copy.deepcopy(self.base_args)
        args.ctc_encoder_layer = 4
        args.reverted_classifier = True
        args.ctc_compress_strategy = "avg"
        args.criterion = "ctc_multi_loss"
        task = test_utils.TestTranslationTask.setup_task(args, self.src_dict, self.tgt_dict)
        task.data_cfg = self.ds.data_cfg
        model = MultitaskConformer.build_model(args, task)
        self.assertFalse(model.auxiliary_decoder.gradient_reversal._lambda_factor.requires_grad)
        self.assertTrue(model.auxiliary_decoder.fc1.weight.requires_grad)
        model.freeze_classifier(update_weights=True)
        self.assertFalse(model.auxiliary_decoder.gradient_reversal._lambda_factor.requires_grad)
        self.assertTrue(model.auxiliary_decoder.fc1.weight.requires_grad)
        model.freeze_classifier()
        self.assertFalse(model.auxiliary_decoder.gradient_reversal._lambda_factor.requires_grad)
        self.assertFalse(model.auxiliary_decoder.fc1.weight.requires_grad)
        model.freeze_classifier(update_weights=True)
        self.assertFalse(model.auxiliary_decoder.gradient_reversal._lambda_factor.requires_grad)
        self.assertTrue(model.auxiliary_decoder.fc1.weight.requires_grad)

    def test_freezing(self):
        args = copy.deepcopy(self.base_args)
        args.ctc_encoder_layer = 4
        args.reverted_classifier = True
        args.ctc_compress_strategy = "avg"
        args.criterion = "ctc_multi_loss"
        task = test_utils.TestTranslationTask.setup_task(args, self.src_dict, self.tgt_dict)
        task.data_cfg = self.ds.data_cfg
        model = MultitaskConformer.build_model(args, task)
        model.freeze_base_model()
        self.assertFalse(model.encoder.conformer_layers[0].conv_module.batchnorm.training)
        model.train()
        self.assertFalse(model.encoder.conformer_layers[0].conv_module.batchnorm.training)
        model.freeze_base_model(update_weights=True)
        self.assertTrue(model.encoder.conformer_layers[0].conv_module.batchnorm.training)


if __name__ == '__main__':
    unittest.main()
