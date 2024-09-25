# Copyright 2024 FBK

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

import torch
from pangolinn import seq2seq
from torch import Tensor, LongTensor, nn

from examples.speech_to_text.models.s2t_transformer_fbk import S2TTransformerEncoder
from examples.speech_to_text.models.s2tlm_transformer_fbk import s2tlm_transformer_s, \
    AudioPrependedTransformerDecoder
from fairseq.data import Dictionary
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models.transformer import Embedding


class AudioPrependedTransformerDecoderWrapper(seq2seq.PangolinnSeq2SeqModuleWrapper):
    def build_module(self) -> nn.Module:
        args = Namespace()
        args.input_feat_per_channel = 4
        args.encoder_embed_dim = 16
        args.input_channels = 1
        args.max_source_positions = 10
        args.max_target_positions = 100
        args.decoder_layers = 4
        args.encoder_layers = 0
        args.criterion = "label_smoothed_cross_entropy"
        args.label_smoothing = 0.1
        args.ctc_compress_strategy = "none"
        s2tlm_transformer_s(args)
        tgt_dict = Dictionary()
        tgt_lines = ["Mi piacciono i quokka", "Mi piacciono le tartarughe", "Mi piacciono gli elefanti"]
        for l in tgt_lines:
            tgt_dict.encode_line(l)

        decoder = AudioPrependedTransformerDecoder(
            args, tgt_dict, Embedding(len(tgt_dict), args.input_feat_per_channel, 2))
        decoder.eval()
        self.encoder = S2TTransformerEncoder(args, tgt_dict)
        self.encoder.eval()
        return decoder

    @property
    def num_input_channels(self) -> int:
        return 1

    @property
    def num_output_channels(self) -> int:
        return 12

    @property
    def input_dtype(self):
        return torch.int

    def forward(self, x: Tensor, lengths: LongTensor) -> Tensor:
        fake_encoder_out = self.encoder(
            torch.ones(x.shape[0], 1, 4).to(x.device),
            torch.ones((x.shape[0])).to(x.device))
        return self._module(x.squeeze(-1), fake_encoder_out)[0]


class S2TLLMTransformerTestCase(seq2seq.CausalTestCase):
    module_wrapper_class = AudioPrependedTransformerDecoderWrapper

    def test_gradient_not_flowing_from_future(self):
        self.skipTest("Integer is not supported")

    def test_incremental_decoding(self):
        """
        Tests that decoding with incremental decoding leads to the same results as doing
        the forward pass without caching previous steps. We perform three steps to make
        sure that it does not work only with one previous step cached.
        """
        decoder = self.module_wrapper.build_module()
        x = torch.randint(10, (2, 10))
        fake_encoder_out = {
            "encoder_out": [torch.ones(3, x.shape[0], 16)],
            "encoder_padding_mask": [lengths_to_padding_mask(torch.LongTensor([3, 2]))],
        }
        incremental_states = {}
        pass1_nostates = decoder(x[:, :1], fake_encoder_out)
        pass1_states = decoder(x[:, :1], fake_encoder_out, incremental_state=incremental_states)
        torch.testing.assert_close(pass1_nostates[0], pass1_states[0])
        pass2_nostates = decoder(x[:, :2], fake_encoder_out)
        pass2_states = decoder(x[:, :2], fake_encoder_out, incremental_state=incremental_states)
        torch.testing.assert_close(pass2_nostates[0][:, -1:, :], pass2_states[0])
        pass3_nostates = decoder(x[:, :3], fake_encoder_out)
        pass3_states = decoder(x[:, :3], fake_encoder_out, incremental_state=incremental_states)
        torch.testing.assert_close(pass3_nostates[0][:, -1:, :], pass3_states[0])

    def test_buffered_mask(self):
        decoder = self.module_wrapper.build_module()
        x = torch.randint(10, (2, 10))
        fake_encoder_out_shorter = {
            "encoder_out": [torch.ones(3, x.shape[0], 16)],
            "encoder_padding_mask": [lengths_to_padding_mask(torch.LongTensor([3, 2]))],
        }
        pass1_shorter = decoder(x, fake_encoder_out_shorter)
        # do 1 pass with longer encoder out so we check that the mask does not remain cached
        # enabling to look in the future
        fake_encoder_out_longer = {
            "encoder_out": [torch.ones(10, x.shape[0], 16)],
            "encoder_padding_mask": [lengths_to_padding_mask(torch.LongTensor([10, 5]))],
        }
        _ = decoder(x, fake_encoder_out_longer)
        pass2_shorter = decoder(x, fake_encoder_out_shorter)
        torch.testing.assert_close(pass1_shorter, pass2_shorter)


if __name__ == '__main__':
    unittest.main()
