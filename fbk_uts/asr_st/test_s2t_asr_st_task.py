# Copyright 2025 FBK
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
from unittest.mock import MagicMock, patch
from argparse import Namespace
import torch
import math

from examples.speech_to_text.tasks.speech_to_text_ctc_asr_st import SpeechToTextCtcASRSTTask


class TestSpeechToTextCtcASRSTTask(unittest.TestCase):
    def setUp(self):
        self.args = Namespace(
            p_sampling_asr=1.0,  # Always ASR
            seed=42,
            data="dummy_data",
            config_yaml="config.yaml"
        )
        self.tgt_dict = MagicMock()
        self.src_dict = MagicMock()
        self.task = SpeechToTextCtcASRSTTask(self.args, self.tgt_dict, self.src_dict)
        self.model = MagicMock()
        self.criterion = MagicMock(return_value=(torch.tensor(1.0), 1, {"sample_size": 1}))
        self.optimizer = MagicMock()

    def test_train_step_asr_mode(self):
        sample = {
            "net_input": {
                "prev_transcript_tokens": torch.tensor([[1, 2]]),
            },
            "prepended_transcript": torch.tensor([[3, 4]]),
            "prepended_transcript_lengths": torch.tensor([2]),
            "ntokens_prepended_transcript": 2
        }

        loss, sample_size, logging_output = self.task.train_step(
            sample, self.model, self.criterion, self.optimizer, update_num=0
        )

        self.assertEqual(sample["target"].tolist(), [[3, 4]])
        self.assertIn("loss", logging_output)
        self.assertIn("sample_size_asr", logging_output)
        self.assertNotIn("sample_size_st", logging_output)

    def test_train_step_st_mode(self):
        st_args = Namespace(
            p_sampling_asr=0.0,  # Always ST
            seed=42,
            data="dummy_data",
            config_yaml="config.yaml"
        )
        st_task = SpeechToTextCtcASRSTTask(st_args, self.tgt_dict, self.src_dict)
        sample = {
            "net_input": {
                "prev_output_tokens": torch.tensor([[1, 2]]),
            },
            "target": torch.tensor([[3, 4]]),
            "target_lengths": torch.tensor([2]),
            "ntokens": 2
        }

        loss, sample_size, logging_output = st_task.train_step(
            sample, self.model, self.criterion, self.optimizer, update_num=0
        )

        self.assertEqual(sample["target"].tolist(), [[3, 4]])
        self.assertIn("loss", logging_output)
        self.assertNotIn("sample_size_asr", logging_output)
        self.assertIn("sample_size_st", logging_output)

    def test_valid_step(self):
        sample = {
            "net_input": {
                "prev_transcript_tokens": torch.tensor([[1, 2]]),
            },
            "prepended_transcript": torch.tensor([[3, 4]]),
            "prepended_transcript_lengths": torch.tensor([2]),
            "ntokens_prepended_transcript": 2
        }

        model = MagicMock()
        criterion = MagicMock(return_value=(torch.tensor(1.0), 1, {"sample_size": 1}))

        loss, sample_size, logging_output = self.task.valid_step(sample, model, criterion)
        self.assertEqual(loss.item(), 2.0)
        self.assertIn("loss", logging_output)
        self.assertIn("sample_size_asr", logging_output)
        self.assertIn("sample_size_st", logging_output)

    @patch("fairseq.metrics.log_scalar")
    def test_reduce_metrics_basic(self, mock_log_scalar):
        logging_outputs = [
            {"loss": 2.0, "sample_size_asr": 2, "sample_size_st": 0, "loss_asr": 2.0}
        ]
        self.task.reduce_metrics(logging_outputs, criterion=None)

        # Check loss and loss_asr to be called as metrics (and equal)
        self.assertAlmostEqual(2.0 / 2 / math.log(2), mock_log_scalar.mock_calls[0][1][1])
        self.assertAlmostEqual(2.0 / 2 / math.log(2), mock_log_scalar.mock_calls[1][1][1])


if __name__ == "__main__":
    unittest.main()
