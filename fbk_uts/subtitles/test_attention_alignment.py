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
import os
import unittest

from examples.speech_to_text.scripts.attention_based_timestamp_estimation import DTWMedianFilterAttentionAligner


class TestAttentionAligners(unittest.TestCase):

    def test_dwt_zero_size_starting_block(self):
        normalized_attn = \
            ("1.5023425565014903,1.5023425565014903,1.5023425565014903,0.6105065962317188,"
             "0.2712440465859149,0.2712440465859149,0.2712440465859149,0.2712440465859149,"
             "0.25956100415460565,-0.28537486330235196 -0.1521569435847247,-0.1521569435847247,"
             "-0.1521569435847247,-0.1188043360943673,-0.1188043360943673,-0.1188043360943673,"
             "-0.1521569435847247,-0.3747284714108167,-0.43193102473556366,-0.43193102473556366 "
             "0.9925778292008444,0.9513012489731852,0.796723439190853,0.796723439190853,"
             "0.9513012489731852,0.9513012489731852,0.796723439190853,-0.0731312496459637,"
             "-0.0731312496459637,-0.0731312496459637 -0.7561187069107445,-0.7561187069107445,"
             "-0.7370387241868251,-0.6860462882165811,-0.6860462882165811,-0.640858725354424,"
             "-0.5383452747882951,-0.4930366632879408,-0.4930366632879408,-0.4930366632879408 "
             "1.5624069635165494,1.5624069635165494,1.5624069635165494,1.5624069635165494,"
             "1.9351700928531435,1.9351700928531435,1.9351700928531435,0.37979772408151913,"
             "0.37979772408151913,0.37979772408151913 -0.9435001470003952,-0.832368533587226,"
             "-0.832368533587226,-0.832368533587226,-0.8789551658755107,-0.8789551658755107,"
             "-0.832368533587226,-0.6414819850834925,-0.6414819850834925,-0.6414819850834925 "
             "-0.854176106101673,-0.8098379679127786,-0.8098379679127786,-0.8098379679127786,"
             "-0.854176106101673,-0.854176106101673,-0.8098379679127786,-0.6636985650436928,"
             "-0.6636985650436928,-0.6636985650436928 -0.3484385719170078,-0.3484385719170078,"
             "-0.28464497041227543,-0.28464497041227543,-0.28464497041227543,-0.3311730092921959,"
             "-0.31954125630721003,-0.31954125630721003,-0.1744218289879788,0.1705200575639368")
        attn_aligner = DTWMedianFilterAttentionAligner(normalized_attn)
        time_idxs = attn_aligner.aligns([3, 7])
        self.assertEqual(len(time_idxs), 2)
        self.assertGreater(time_idxs[0], 0)
        self.assertGreater(time_idxs[1], time_idxs[0])

    def dwt_assert_nonzero_block(self, attn_file, boundaries):
        path = os.path.join(os.path.dirname(__file__), attn_file)
        with open(path) as f:
            norm_attn = f.read().strip()
        attn_aligner = DTWMedianFilterAttentionAligner(norm_attn)
        time_idxs = attn_aligner.aligns(boundaries)
        self.assertEqual(len(time_idxs), len(boundaries))
        self.assertGreater(time_idxs[0], 0)
        for i in range(len(boundaries) - 1):
            self.assertGreater(time_idxs[i + 1], time_idxs[i])

    def test_dwt_zero_size_not_starting_block(self):
        self.dwt_assert_nonzero_block("resources/attn_string_sample.txt", [3, 8, 12])

    def test_dwt_zero_size_not_starting_block_2(self):
        self.dwt_assert_nonzero_block("resources/attn_string_sample_2.txt", [3, 8, 18])


if __name__ == '__main__':
    unittest.main()
