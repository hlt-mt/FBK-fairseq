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
from unittest.mock import patch
import tests.utils

import torch
import copy
from argparse import Namespace

from examples.speech_to_text.data.speech_to_text_dataset_with_src import SpeechToTextDatasetWithSrc
from examples.speech_to_text.models.s2t_transformer_fbk import s2t_transformer_s, S2TTransformerModel
from fairseq import checkpoint_utils
from fbk_uts.base_utilities import BaseSpeechTestCase


class UpdateWeightsToExtraTokensTestCase(unittest.TestCase, BaseSpeechTestCase):

    def setUp(self):
        self.init_sample_dataset(SpeechToTextDatasetWithSrc)
        args = Namespace()
        args.encoder_embed_dim = 16
        args.input_feat_per_channel = 5
        args.input_channels = 1
        args.max_source_positions = 10
        args.max_target_positions = 10
        args.encoder_layers = 2
        args.criterion = "label_smoothed_crossentropy"
        args.ctc_compress_strategy = "none"
        args.init_variance = 5.0
        s2t_transformer_s(args)
        self.task = tests.utils.TestTranslationTask.setup_task(
            args, self.src_dict, self.tgt_dict)
        self.args = args

    @patch('fairseq.models.fairseq_model.BaseFairseqModel.upgrade_state_dict_named')
    def test_no_allow_extra_tokens_argument(self, mock_upgrade_state_dict_fairseq):
        mock_upgrade_state_dict_fairseq.return_value = None
        model = S2TTransformerModel.build_model(self.args, self.task)
        state_dict = {
            "decoder.embed_tokens.weight": torch.rand((10, 4)),
            "decoder.output_projection.weight": torch.rand((10, 4))}
        # Control that the following has not changed the dimension of the weights
        # matrix since allow_extra_tokens is disabled.
        model.upgrade_state_dict_named(state_dict, None)
        self.assertEqual(len(model.decoder.dictionary), 12)
        self.assertEqual(list(state_dict["decoder.embed_tokens.weight"].shape), [10, 4])
        self.assertEqual(list(state_dict["decoder.output_projection.weight"].shape), [10, 4])

    @patch('fairseq.models.fairseq_model.BaseFairseqModel.upgrade_state_dict_named')
    def test_allow_extra_tokens(self, mock_upgrade_state_dict_fairseq):
        mock_upgrade_state_dict_fairseq.return_value = None
        new_args = copy.deepcopy(self.args)
        new_args.allow_extra_tokens = True
        model = S2TTransformerModel.build_model(new_args, self.task)
        state_dict = {
            "decoder.embed_tokens.weight": torch.rand((10, 4)),
            "decoder.output_projection.weight": torch.rand((10, 4))}
        # The following method changes the both weights in state_dict
        model.upgrade_state_dict_named(state_dict, None)
        self.assertEqual(len(model.decoder.dictionary), 12)
        self.assertEqual(list(state_dict["decoder.embed_tokens.weight"].shape), [12, 4])
        self.assertEqual(list(state_dict["decoder.output_projection.weight"].shape), [12, 4])

    def test_only_decoder_embed_tokens_changed(self):
        new_args = copy.deepcopy(self.args)
        new_args.allow_extra_tokens = True
        model = S2TTransformerModel.build_model(new_args, self.task)
        state_dict = {
            "decoder.embed_tokens.weight": torch.rand((10, 4)),
            "decoder.output_projection.weight": torch.rand((10, 4))}
        # The following must update only the named weights passed as argument
        # that is embed_tokens and not the output_projection ones.
        checkpoint_utils.update_weights_to_extra_tokens(
            model.decoder, "decoder.embed_tokens.weight", state_dict)
        self.assertEqual(len(model.decoder.dictionary), 12)
        self.assertEqual(list(state_dict["decoder.embed_tokens.weight"].shape), [12, 4])
        self.assertEqual(list(state_dict["decoder.output_projection.weight"].shape), [10, 4])


if __name__ == '__main__':
    unittest.main()
