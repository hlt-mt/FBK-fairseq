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
import torch
import copy

from examples.speech_to_text.criterions.ctc_multi_loss import CTCMultiLoss
from examples.speech_to_text.data.speech_to_text_dataset_with_src import SpeechToTextDatasetWithSrc
from examples.speech_to_text.models.conformer import ConformerModel, conformer_s
from examples.speech_to_text.models.s2t_transformer_fbk import S2TTransformerEncoder, S2TTransformerModel, \
    s2t_transformer_s
from fairseq import optim
from fbk_uts.base_utilities import BaseSpeechTestCase
from tests.utils import TestTranslationTask


class S2TTransformerEncoderTestCase(unittest.TestCase, BaseSpeechTestCase):
    @patch('fairseq.data.audio.speech_to_text_dataset.get_features_or_waveform')
    def setUp(self, mock_get_features_or_waveform) -> None:
        mock_get_features_or_waveform.return_value = np.random.random((20, 4))
        self.init_sample_dataset(SpeechToTextDatasetWithSrc)
        args = Namespace()
        s2t_transformer_s(args)
        args.input_feat_per_channel = 4
        args.input_channels = 1
        args.max_source_positions = 10
        args.max_target_positions = 100
        args.encoder_layers = 2
        args.decoder_layers = 2
        args.underlying_criterion = "label_smoothed_cross_entropy"
        args.criterion = "ctc_multi_loss"
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
        args.data = "dummy"
        args.config_yaml = "dummy"
        args.lr = [1e-3]
        args.momentum = 0
        args.weight_decay = 0
        self.args = args
        self.task = TestTranslationTask.setup_task(
            self.args, self.src_dict, self.tgt_dict)
        self.samples = self.ds.collater([self.ds[0], self.ds[1], self.ds[2]])

    def train_step(self, args, model):
        # Train step over the model to update weights
        model.train()
        criterion = CTCMultiLoss(args, self.task)
        loss, _, _ = criterion.forward(model, self.samples)
        optimizer = optim.sgd.SGD(args, model.parameters())
        optimizer.backward(loss)
        optimizer.step()

    def test_subsample_with_padding(self):
        encoder = S2TTransformerEncoder(self.args, self.src_dict)
        encoder.eval()
        torch.manual_seed(0)
        fake_sample = torch.rand(2, 40, 4)
        fake_sample[1, 20:, :] = 0
        fake_lengths = torch.LongTensor([40, 20])
        fake_sample2 = fake_sample[1:, :20, :]
        subsample_out = encoder.subsample(fake_sample, fake_lengths)
        subsample_out2 = encoder.subsample(fake_sample2, fake_lengths[1:])
        # Sum of padded elements must be equal to 0
        self.assertTrue((subsample_out[0][5:, 1, :].sum() == 0).item())
        # Non padded elements of the processed batch must be equal to the unpadded sample
        torch.testing.assert_allclose(
            subsample_out[0][:5, 1, :],
            subsample_out2[0][:, 0, :])
        encoder_out = encoder(fake_sample, fake_lengths)
        encoder_out2 = encoder(fake_sample2, fake_lengths[1:])
        # Ensure encoder output
        torch.testing.assert_allclose(
            encoder_out['encoder_out'][0][:5, 1, :],
            encoder_out2['encoder_out'][0][:, 0, :])

    def test_ctc(self):
        model = S2TTransformerModel.build_model(self.args, self.task)
        criterion = CTCMultiLoss(self.args, self.task)
        loss, _, _ = criterion.forward(model, self.samples)
        self.assertTrue(loss > 0)

    def test_freeze_transformer_encoder(self):
        new_args = copy.deepcopy(self.args)
        new_args.freeze_encoder = True
        model = S2TTransformerModel.build_model(new_args, self.task)
        for _, param in model.encoder.named_parameters():
            self.assertFalse(param.requires_grad)
        # Save encoder parameters to check later to be unaltered by the train step
        encoder_before_forward = copy.deepcopy(model.encoder)
        self.train_step(new_args, model)
        encoder_state_dict = model.encoder.state_dict()
        for name, param in encoder_before_forward.named_parameters():
            self.assertTrue(torch.equal(param, encoder_state_dict[name]))

    def test_freeze_conformer_encoder(self):
        new_args = copy.deepcopy(self.args)
        conformer_s(new_args)
        new_args.freeze_encoder = True
        new_args.no_syncbatchnorm = True
        model = ConformerModel.build_model(new_args, self.task)
        for _, param in model.encoder.named_parameters():
            self.assertFalse(param.requires_grad)
        # Check BatchNorm
        self.assertFalse(model.encoder.conformer_layers[0].conv_module.batchnorm.training)
        # Save encoder parameters to check later to be unaltered by the train step
        encoder_before_forward = copy.deepcopy(model.encoder)
        self.train_step(new_args, model)
        encoder_state_dict = model.encoder.state_dict()
        for name, param in encoder_before_forward.named_parameters():
            self.assertTrue(torch.equal(param, encoder_state_dict[name]))
        # Check BatchNorm statistics
        for enc_lay in range(2):
            self.assertTrue(torch.equal(
                model.encoder.conformer_layers[enc_lay].conv_module.batchnorm.running_mean,
                encoder_before_forward.conformer_layers[enc_lay].conv_module.batchnorm.running_mean))
            self.assertTrue(torch.equal(
                model.encoder.conformer_layers[enc_lay].conv_module.batchnorm.running_var,
                encoder_before_forward.conformer_layers[enc_lay].conv_module.batchnorm.running_var))
            self.assertTrue(torch.equal(
                model.encoder.conformer_layers[enc_lay].conv_module.batchnorm.num_batches_tracked,
                encoder_before_forward.conformer_layers[enc_lay].conv_module.batchnorm.num_batches_tracked))


if __name__ == '__main__':
    unittest.main()
