# Copyright 2021 FBK

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

from examples.speech_to_text.criterions.cross_entropy_dualdecoder_with_tags import masked_label_smoothed_ce


class CrossEntropyDualdecWithTagsTestCase(unittest.TestCase):
    def test_basic(self):
        lprobs = torch.log(
            torch.FloatTensor([
                [
                    [0.8, 0.05, 0.05, 0.1],
                    [0.6, 0.15, 0.2, 0.05],
                    [0.1, 0.75, 0.1, 0.05],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.65, 0.10, 0.2, 0.05],
                    [0.05, 0.10, 0.8, 0.05],
                    [0.05, 0.7, 0.2, 0.05],
                    [0.35, 0.10, 0.3, 0.25],
                    [0.05, 0.70, 0.2, 0.05],
                ],
            ]),
        )
        target_tags = torch.LongTensor([[0, 0, 1, 0, 0], [0, 2, 1, 0, 1]])
        target = torch.LongTensor([[5, 5, 5, 1, 1], [4, 5, 6, 5, 5]])
        loss, nll_loss = masked_label_smoothed_ce(
            lprobs.view(-1, lprobs.size(-1)),
            target_tags.view(-1, 1),
            0.1,
            pad_mask=target.view(-1, 1).eq(1),
            reduce=False
        )
        for i in range(10):
            if i == 3 or i == 4:
                self.assertEqual(0.0, loss[i])
            else:
                self.assertTrue(0.0 < loss[i])


if __name__ == '__main__':
    unittest.main()
