# Copyright 2022 FBK
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

import torch
import copy
import numpy as np

import tests.utils
from examples.speech_to_text.criterions.ctc_multi_loss import CTCMultiLoss
from examples.speech_to_text.data.speech_to_text_dataset_multimodal import SpeechToTextDatasetMultimodal
from examples.speech_to_text.inference.sequence_generator_dual_encoder import DualEncoderSequenceGenerator
from examples.speech_to_text.models.s2t_transformer_fbk_triangle import s2t_transformer_triangle_s
from examples.speech_to_text.models.s2t_transformer_dual_encoder import S2TTransformerDualEncoderModel
from fbk_uts.base_utilities import BaseSpeechTestCase


class DualEncoderTestCase(unittest.TestCase, BaseSpeechTestCase):
    @patch('fairseq.data.audio.speech_to_text_dataset.get_features_or_waveform')
    def setUp(self, mock_get_features_or_waveform) -> None:
        mock_get_features_or_waveform.return_value = np.random.random((20, 4))
        self.init_sample_dataset(SpeechToTextDatasetMultimodal)
        args = Namespace()
        args.underlying_criterion = "label_smoothed_cross_entropy"
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
        args.ctc_weight = 0.5
        s2t_transformer_triangle_s(args)
        args.encoder_layers = 4
        args.decoder_layers = 2
        args.context_encoder_layers = 2
        args.input_feat_per_channel = 4
        args.input_channels = 1
        args.max_source_positions = 10
        args.context_dropout = 0.3
        args.encoder_embed_dim = 256
        args.decoder_embed_dim = 256
        args.share_encoder_decoder_embed = False
        args.context_ffn_embed_dim = 1024
        args.context_decoder_attention_type = "parallel"
        args.pretrained_model = None
        self.args = args
        self.task = tests.utils.TestTranslationTask.setup_task(
            self.args, self.src_dict, self.tgt_dict
        )
        self.samples = self.ds.collater([self.ds[0], self.ds[1], self.ds[2]])

    def test_context_encoder(self):
        model = S2TTransformerDualEncoderModel.build_model(self.args, self.task)
        self.assertEqual(len(model.context_encoder.layers), self.args.context_encoder_layers)

    def test_dual_encoder_parallel_forward_ctc(self):
        model = S2TTransformerDualEncoderModel.build_model(self.args, self.task)
        criterion = CTCMultiLoss(self.args, self.task)
        loss, _, _ = criterion.forward(model, self.samples)
        self.assertTrue(loss > 0)

    def test_dual_encoder_sequential_forward_ctc(self):
        args2 = copy.deepcopy(self.args)
        args2.context_decoder_attention_type = "sequential"
        model = S2TTransformerDualEncoderModel.build_model(args2, self.task)
        criterion = CTCMultiLoss(args2, self.task)
        loss, _, _ = criterion.forward(model, self.samples)
        self.assertTrue(loss > 0)

    def test_dual_encoder_forward(self):
        # Check if model forward outputs a torch.Tensor and not None during training
        model = S2TTransformerDualEncoderModel.build_model(self.args, self.task)
        forward_output = model.forward(**self.samples["net_input"])
        self.assertTrue(isinstance(forward_output[0][0], torch.Tensor))
        logprobs = model.get_normalized_probs(forward_output[0], log_probs=True)
        self.assertTrue(isinstance(logprobs, torch.Tensor))

    def test_dual_encoder_eval(self):
        # Check if model forward outputs a torch.Tensor and not None during evaluation
        model = S2TTransformerDualEncoderModel.build_model(self.args, self.task)
        # Enable code parts which are ignored during training
        model.eval()
        forward_output = model.forward(**self.samples["net_input"])
        self.assertTrue(isinstance(forward_output[0][0], torch.Tensor))
        logprobs = model.get_normalized_probs(forward_output[0], log_probs=True)
        self.assertTrue(isinstance(logprobs, torch.Tensor))

    def test_generate(self):
        model = S2TTransformerDualEncoderModel.build_model(self.args, self.task)
        generator = DualEncoderSequenceGenerator(
            [model], self.tgt_dict, beam_size=2, normalize_scores=False
        )
        hypos = generator.forward(self.samples)
        self.assertTrue(isinstance(hypos[0][0]["tokens"], torch.LongTensor))


if __name__ == '__main__':
    unittest.main()
