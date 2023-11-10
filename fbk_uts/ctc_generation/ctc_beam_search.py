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

import torch

from examples.speech_to_text.inference.ctc_generator import CTCBeamEntry
from examples.speech_to_text.tasks.speech_to_text_ctcgen import CTCGenerator


class BeamSearchTestCase(unittest.TestCase):
    def test_deduplicate(self):
        beam = [
            CTCBeamEntry(
                [],
                torch.tensor(float('-inf')),
                torch.tensor(float('-inf'))),
            CTCBeamEntry(
                [1, 2],
                torch.tensor(-1.0),
                torch.tensor(-2.0)),
            CTCBeamEntry(
                [1],
                torch.tensor(-1.0),
                torch.tensor(-1.0)),
            CTCBeamEntry(
                [],
                torch.tensor(float('-inf')),
                torch.tensor(-1.0)),
            CTCBeamEntry(
                [1, 2],
                torch.tensor(-1.0),
                torch.tensor(-0.5)),
        ]
        dedup_beam = CTCGenerator.deduplicate(beam)
        self.assertEqual(3, len(dedup_beam))
        dedup_beam = sorted(dedup_beam, key=lambda x: x.prefix)
        self.assertEqual([], dedup_beam[0].prefix)
        self.assertEqual(float('-inf'), dedup_beam[0].non_blank_end_logprob)
        self.assertEqual(-1.0, dedup_beam[0].blank_end_logprob)


if __name__ == '__main__':
    unittest.main()
