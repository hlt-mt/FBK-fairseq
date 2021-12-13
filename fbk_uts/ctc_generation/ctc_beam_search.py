import unittest

import torch

from examples.speech_to_text.tasks.speech_to_text_ctcgen import CTCGenerator, CTCBeamEntry


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
