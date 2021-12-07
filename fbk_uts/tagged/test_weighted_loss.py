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
import argparse
import unittest

import torch

from examples.speech_to_text.criterions.weighted_label_smoothed_cross_entropy import \
    WeightedLabelSmoothedCrossEntropyCriterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion
from tests import utils as test_utils


class WeightedLossTestCase(unittest.TestCase):
    def setUp(self):
        # build dictionary
        self.d = test_utils.dummy_dictionary(3)
        vocab = len(self.d)
        self.assertEqual(vocab, 4 + 3)  # 4 special + 3 tokens
        self.assertEqual(self.d.pad(), 1)
        self.assertEqual(self.d.eos(), 2)
        self.assertEqual(self.d.unk(), 3)
        # build model
        self.args = argparse.Namespace()
        self.args.probs = []
        self.task = test_utils.TestTranslationTask.setup_task(self.args, self.d, self.d)
        self.model = self.task.build_model(self.args)

    def test_basic_usage(self):
        loss = WeightedLabelSmoothedCrossEntropyCriterion(
            self.task,
            sentence_avg=True,
            label_smoothing=0.1,
            tag_weight=2.0,
            notag_weight=1.0,
        )
        basic_loss = LabelSmoothedCrossEntropyCriterion(
            self.task,
            sentence_avg=True,
            label_smoothing=0.1,
        )
        net_output = (
            torch.FloatTensor([
                [
                    #      pad   eos  unk   w1   w2   w3
                    [0.05, 0.05, 0.1, 0.05, 0.3, 0.4, 0.05],
                    [0.05, 0.10, 0.2, 0.05, 0.2, 0.3, 0.10],
                    [0.05, 0.15, 0.3, 0.05, 0.1, 0.2, 0.15],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                [
                    #      pad   eos  unk   w1   w2   w3
                    [0.05, 0.10, 0.2, 0.05, 0.2, 0.3, 0.10],
                    [0.05, 0.10, 0.2, 0.05, 0.2, 0.3, 0.10],
                    [0.05, 0.10, 0.2, 0.05, 0.2, 0.3, 0.10],
                    [0.05, 0.10, 0.2, 0.05, 0.2, 0.3, 0.10],
                    [0.05, 0.10, 0.2, 0.05, 0.2, 0.3, 0.10],
                ],
            ]),
        )
        sample = {
            "target": torch.LongTensor([[5, 5, 5, 1, 1], [4, 5, 6, 5, 5]]),
            "target_tags": torch.LongTensor([[0, 1, 0, 0, 0], [12, 7, 0, 0, 0]]),
        }
        loss, nll_loss = loss.compute_loss(self.model, net_output, sample, reduce=False)
        basic_loss, basic_nll_loss = basic_loss.compute_loss(self.model, net_output, sample, reduce=False)
        self.assertEqual(loss.shape, basic_loss.shape)
        for i in [1, 5, 6]:
            self.assertEqual(loss[i][0].item(), basic_loss[i][0].item()*2)
        for i in range(10):
            if i in {1, 5, 6}:
                self.assertEqual(loss[i][0].item(), basic_loss[i][0].item() * 2)
            else:
                self.assertEqual(loss[i][0].item(), basic_loss[i][0].item())


if __name__ == '__main__':
    unittest.main()
