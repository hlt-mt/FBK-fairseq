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

from examples.speech_to_text.occlusion_explanation.perturbators import PERTURBATION_REGISTRY, \
    OcclusionDecoderEmbeddingsPerturbator


class TestPerturbators(unittest.TestCase):

    # assert no exception is raised when creating a class with valid probability values
    def test_from_config_dict_allowed_p(self):
        for p in [0.0, 0.1, 0.9]:
            for name, clazz in PERTURBATION_REGISTRY.items():
                if issubclass(clazz, OcclusionDecoderEmbeddingsPerturbator):
                    config_name = "decoder_occlusion"
                else:
                    config_name = "fbank_occlusion"
                try:
                    _ = clazz.from_config_dict(config={config_name: {"p": p}})
                except AssertionError:
                    raise AssertionError(f"{p} not allowed in {name}")

    # assert exception is raised when creating a class with invalid probability values
    def test_from_config_dict_not_allowed_p(self):
        for p in [-0.1, 1.1, 1.0]:
            for name, clazz in PERTURBATION_REGISTRY.items():
                if issubclass(clazz, OcclusionDecoderEmbeddingsPerturbator):
                    config_name = "decoder_occlusion"
                else:
                    config_name = "fbank_occlusion"
                try:
                    with self.assertRaises(AssertionError):
                        _ = clazz.from_config_dict(config={config_name: {"p": p}})
                except AssertionError:
                    raise AssertionError(f"{p} allowed in {name}")


if __name__ == '__main__':
    unittest.main()
