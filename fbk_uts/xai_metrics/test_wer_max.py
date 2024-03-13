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

from fairseq.scoring.wer_max import WerMaxValueClippingScorer


class MockWerValueClippingScorer(WerMaxValueClippingScorer):
    def __init__(self, distance, ref_length):
        self.distance = distance
        self.ref_length = ref_length


class TesWerMaxScorer(unittest.TestCase):
    def test_score(self):
        scorer1 = MockWerValueClippingScorer(100, 20)
        scorer2 = MockWerValueClippingScorer(3, 20)
        scorer3 = MockWerValueClippingScorer(20, 20)
        self.assertEqual(scorer1.score(), 100)
        self.assertEqual(scorer2.score(), 15)
        self.assertEqual(scorer3.score(), 100)


if __name__ == '__main__':
    unittest.main()
