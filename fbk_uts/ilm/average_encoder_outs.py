import argparse
import unittest

import torch
import numpy as np

from examples.speech_to_text.inference.average_encoder_outputs import EncoderStatesAverage
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models import FairseqEncoder, FairseqEncoderDecoderModel, FairseqIncrementalDecoder
from tests.utils import TestTranslationTask


class TestEncoder(FairseqEncoder):
    def __init__(self, args, dictionary):
        super().__init__(dictionary)
        self.args = args

    def forward(self, src_tokens, src_lengths=None, **kwargs):
        encoder_padding_mask = lengths_to_padding_mask(src_lengths)
        return {
            "encoder_out": [src_tokens],
            "encoder_padding_mask": [encoder_padding_mask] if encoder_padding_mask.any() else [],
            "encoder_embedding": [],
            "encoder_states": [],
            "src_tokens": [],
            "src_lengths": [],
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        new_encoder_out = (
            [] if len(encoder_out["encoder_out"]) == 0
            else [x.index_select(1, new_order) for x in encoder_out["encoder_out"]]
        )

        new_encoder_padding_mask = (
            [] if len(encoder_out["encoder_padding_mask"]) == 0
            else [x.index_select(0, new_order) for x in encoder_out["encoder_padding_mask"]]
        )
        return {
            "encoder_out": new_encoder_out,
            "encoder_padding_mask": new_encoder_padding_mask,
            "encoder_embedding": [],
            "encoder_states": [],
            "src_tokens": [],
            "src_lengths": [],
        }


class FakeDecoder(FairseqIncrementalDecoder):
    def __init__(self, args, dictionary):
        super().__init__(dictionary)
        self.args = args

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None):
        return None

    def get_normalized_probs(self, net_output, log_probs, _):
        # the decoder returns probabilities directly
        probs = net_output[0]
        if log_probs:
            return probs.log()
        else:
            return probs


class TestModel(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @classmethod
    def build_model(cls, args, task):
        encoder = TestEncoder(args, task.source_dictionary)
        decoder = FakeDecoder(args, task.target_dictionary)
        return cls(encoder, decoder)


class EncoderAverageTestCase(unittest.TestCase):
    def setUp(self) -> None:
        args = argparse.Namespace()
        task = TestTranslationTask.setup_task(args)
        self.model = TestModel.build_model(args, task)

    def test_base(self):
        # T x B x C --- 5 x 3 x 4
        easy_batch_data = torch.Tensor([
            [[0.1, 12.3, -0.6, 1.0], [-0.1, -0.3, 1.4, 0.2], [-0.6, 0.1, 0.2, 0.3]],
            [[0.5, 0.4, -0.3, -11.2], [-0.2, -1.3, 0.2, 0.7], [0., 0., 0., 0.]],
            [[0.8, 0.9, 0.1, 0.4], [0.1, 0.1, 0.1, 0.1], [0., 0., 0., 0.]],
            [[0., -2.0, -1.5, 0.4], [0., 0., 0., 0.], [0., 0., 0., 0.]],
            [[1.0, 0.3, 0.6, -0.9], [0., 0., 0., 0.], [0., 0., 0., 0.]]
        ])
        base_sample = {'net_input': {'src_tokens': easy_batch_data, 'src_lengths': torch.LongTensor([5, 3, 1])}}
        averager = EncoderStatesAverage(self.model)
        averager.add_batch(base_sample)
        self.assertEqual(averager.num_encoder_outs, 9)
        self.assertTrue(np.allclose(averager.current_value, np.array([1.6 / 9, 10.5 / 9, 0.2 / 9, -1.0])))
        averager.add_batch(base_sample)
        self.assertEqual(averager.num_encoder_outs, 18)
        self.assertTrue(np.allclose(averager.current_value, np.array([1.6 / 9, 10.5 / 9, 0.2 / 9, -1.0])))

    def test_no_padding(self):
        easy_batch_data = torch.Tensor([
            [[0.1, 12.3, -0.6, 1.0], [-0.1, -0.3, 1.4, 0.2], [-0.6, 0.1, 0.2, 0.3]],
            [[0.5, 0.4, -0.3, -11.2], [-0.2, -1.3, 0.2, 0.7], [0.1, 0.1, 0., 0.2]],
        ])
        base_sample = {'net_input': {'src_tokens': easy_batch_data, 'src_lengths': torch.LongTensor([2, 2, 2])}}
        averager = EncoderStatesAverage(self.model)
        averager.add_batch(base_sample)
        self.assertEqual(averager.num_encoder_outs, 6)
        self.assertTrue(np.allclose(averager.current_value, np.array([-0.2 / 6, 11.3 / 6, 0.9 / 6, -8.8 / 6])))

    def test_padding_nonzero(self):
        # T x B x C --- 5 x 3 x 4
        easy_batch_data = torch.Tensor([
            [[0.1, 12.3, -0.6, 1.0], [-0.1, -0.3, 1.4, 0.2], [-0.6, 0.1, 0.2, 0.3]],
            [[0.5, 0.4, -0.3, -11.2], [-0.2, -1.3, 0.2, 0.7], [1., 1., 1., 1.]],
            [[0.8, 0.9, 0.1, 0.4], [0.1, 0.1, 0.1, 0.1], [1., 1., 1., 1.]],
            [[0., -2.0, -1.5, 0.4], [0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]],
            [[1.0, 0.3, 0.6, -0.9], [0.1, 0.1, 0.1, 0.1], [100., -10000., -1., 1000.]]
        ])
        base_sample = {'net_input': {'src_tokens': easy_batch_data, 'src_lengths': torch.LongTensor([5, 3, 1])}}
        averager = EncoderStatesAverage(self.model)
        averager.add_batch(base_sample)
        self.assertEqual(averager.num_encoder_outs, 9)
        self.assertTrue(np.allclose(averager.current_value, np.array([1.6 / 9, 10.5 / 9, 0.2 / 9, -1.0])))
        averager.add_batch(base_sample)
        self.assertEqual(averager.num_encoder_outs, 18)
        self.assertTrue(np.allclose(averager.current_value, np.array([1.6 / 9, 10.5 / 9, 0.2 / 9, -1.0])))

if __name__ == '__main__':
    unittest.main()
