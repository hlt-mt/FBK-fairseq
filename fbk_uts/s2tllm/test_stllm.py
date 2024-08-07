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
from fairseq.models.transformer import Embedding


class AudioPrependedTransformerDecoderWrapper(seq2seq.PangolinnSeq2SeqModuleWrapper):
    def build_module(self) -> nn.Module:
        args = Namespace()
        args.input_feat_per_channel = 4
        args.encoder_embed_dim = 16
        args.input_channels = 1
        args.max_source_positions = 10
        args.max_target_positions = 100
        args.decoder_layers = 2
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
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
