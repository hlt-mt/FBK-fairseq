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

from torch import nn, Tensor, LongTensor

from examples.speech_to_text.modules.hyena import HyenaOperator
from fairseq.data.data_utils import lengths_to_padding_mask

from pangolinn import seq2seq


class HyenaOperatorPangolinnWrapper(seq2seq.PangolinnSeq2SeqModuleWrapper):
    def build_module(self) -> nn.Module:
        return HyenaOperator(self.num_input_channels, 30, num_heads=4)

    @property
    def num_input_channels(self) -> int:
        return 8

    def forward(self, x: Tensor, lengths: LongTensor) -> Tensor:
        return self._module(x, lengths_to_padding_mask(lengths))


class HyenaNonCausalOperatorPangolinnWrapper(HyenaOperatorPangolinnWrapper):
    def build_module(self) -> nn.Module:
        return HyenaOperator(self.num_input_channels, 30, num_heads=4, causal=False)


class HyenaOperatorPaddingTestCase(seq2seq.EncoderPaddingTestCase):
    module_wrapper_class = HyenaOperatorPangolinnWrapper


class HyenaNonCausalOperatorPaddingTestCase(seq2seq.EncoderPaddingTestCase):
    module_wrapper_class = HyenaNonCausalOperatorPangolinnWrapper


class HyenaOperatorCausalityTestCase(seq2seq.CausalTestCase):
    module_wrapper_class = HyenaOperatorPangolinnWrapper


class HyenaNonCausalOperatorCausalityTestCase(seq2seq.CausalTestCase):
    module_wrapper_class = HyenaNonCausalOperatorPangolinnWrapper

    def test_not_looking_at_the_future(self):
        with self.assertRaises(AssertionError):
            super().test_not_looking_at_the_future()

    def test_gradient_not_flowing_from_future(self):
        with self.assertRaises(AssertionError):
            super().test_gradient_not_flowing_from_future()


if __name__ == '__main__':
    unittest.main()
