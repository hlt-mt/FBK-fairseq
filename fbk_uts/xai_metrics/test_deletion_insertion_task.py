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

import argparse
import os

import unittest
import torch

from examples.speech_to_text.tasks.deletion_insertion_task_ctc import FeatureAttributionEvaluationCtcTask
from examples.speech_to_text.tasks.deletion_insertion_task import FeatureAttributionEvaluationTask
from fairseq.tasks import TASK_REGISTRY, FairseqTask


class TestDeletionInsertionTask(unittest.TestCase):
    def setUp(self):
        args = argparse.Namespace()
        args.data = "path/to/data"
        args.config_yaml = "config.yaml"
        args.perc_interval = 5
        args.metric = "deletion"
        args.aggregator = "sentence"
        args.normalizer = []
        current_directory = os.path.dirname(__file__)
        relative_path = os.path.join('mock_data', 'explanations.h5')
        explanation_path = os.path.join(current_directory, relative_path)
        args.explanation_path = explanation_path
        self.task = FeatureAttributionEvaluationTask(args=args, tgt_dict=None)
        self.task_ctc = FeatureAttributionEvaluationCtcTask(args=args, tgt_dict=None, src_dict=None)

    def test_task_registration(self):
        self.assertIn("feature_attribution_evaluation_task", TASK_REGISTRY)
        task_cls = TASK_REGISTRY["feature_attribution_evaluation_task"]
        self.assertTrue(issubclass(task_cls, FairseqTask))
    
    def test_task_registration(self):
        self.assertIn("feature_attribution_evaluation_task_ctc", TASK_REGISTRY)
        task_cls = TASK_REGISTRY["feature_attribution_evaluation_task_ctc"]
        self.assertTrue(issubclass(task_cls, FairseqTask))

    def test_customize_sample_id(self):
        sample = {
            "id": torch.tensor([
                20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61]),
            "orig_id": torch.tensor([
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])}
        new_sample_id = self.task.customize_sample_id(24, 44, sample)
        self.assertEqual(new_sample_id, "1-2")
        new_sample_id = self.task_ctc.customize_sample_id(24, 44, sample)
        self.assertEqual(new_sample_id, "1-2")

    def test_assertion_interval_size(self):
        args = argparse.Namespace()
        args.data = "path/to/data"
        args.config_yaml = "config.yaml"
        args.perc_interval = 7
        args.metric = "deletion"
        args.aggregator = "sentence"
        args.normalizer = []
        current_directory = os.path.dirname(__file__)
        relative_path = os.path.join('mock_data', 'explanations.h5')
        explanation_path = os.path.join(current_directory, relative_path)
        args.explanation_path = explanation_path
        with self.assertRaises(AssertionError):
            metric_dataset_with_src = FeatureAttributionEvaluationCtcTask(
                args=args, tgt_dict=None, src_dict=None)
        with self.assertRaises(AssertionError):
            metric_dataset = FeatureAttributionEvaluationTask(args=args, tgt_dict=None)


if __name__ == '__main__':
    unittest.main()
