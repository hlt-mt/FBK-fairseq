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
from unittest.mock import patch

from examples.speech_to_text.simultaneous_translation.scripts.stream_laal import (
    parse_simuleval_instances, parse_references, resegment_instances, SegmentLevelDelayElapsed, SimulEvalLogInstance)


class StreamLAALTest(unittest.TestCase):
    def setUp(self):
        self.instances = parse_simuleval_instances("./resources/instances.log", "word")
        self.references = parse_references(
            "./resources/references.txt", "./resources/audio_info.yaml")
        self.resegmented_preds = [
            "Ich brauche , dass du nach oben gehst, das Feuer vorbeigehst,",
            "und ich brauche, dass du diese Frau ein Paar Schuhe holst. (Gel\u00e4chter)"]

    def test_parse_simuleval_instances(self):
        self.assertEqual(list(self.instances.keys()), ["ted_1096.wav"])
        self.assertEqual(
            self.instances["ted_1096.wav"].prediction,
            "Ich brauche , dass du nach oben gehst, das Feuer vorbeigehst, und ich brauche, dass"
            " du diese Frau ein Paar Schuhe holst. (Gel\u00e4chter)")
        self.assertEqual(
            self.instances["ted_1096.wav"].reference,
            "Sie m\u00fcssen nach oben gehen, an dem Feuer vorbei, und m\u00fcssen dieser Frau "
            "ein Paar Schuhe holen. \"(Gel\u00e4chter) Ich schw\u00f6re es.")
        self.assertEqual(self.instances["ted_1096.wav"].latency_unit, "word")

    def test_parse_references(self):
        self.assertEqual(list(self.references.keys()), ["ted_1096.wav"])
        self.assertEqual(
            self.references["ted_1096.wav"][0].content,
            "Sie m\u00fcssen nach oben gehen, an dem Feuer vorbei,")
        self.assertEqual(
            self.references["ted_1096.wav"][1].content,
            "und m\u00fcssen dieser Frau ein Paar Schuhe holen. \"(Gel\u00e4chter) Ich "
            "schw\u00f6re es.")
        self.assertEqual(self.references["ted_1096.wav"][0].duration, 4.0)
        self.assertEqual(self.references["ted_1096.wav"][0].start_time, 0.0)
        self.assertEqual(self.references["ted_1096.wav"][1].duration, 5.34)
        self.assertEqual(self.references["ted_1096.wav"][1].start_time, 4.0)

    @patch(
        'examples.speech_to_text.simultaneous_translation.scripts.stream_laal.MwerSegmenter.'
        '__init__')
    @patch(
        'examples.speech_to_text.simultaneous_translation.scripts.stream_laal.MwerSegmenter.'
        '__call__')
    def test_resegment_instances(self, mock_mwer_call, mock_mwer_init):
        mock_mwer_init.return_value = None
        mock_mwer_call.return_value = self.resegmented_preds
        resegmented_instances = resegment_instances(self.instances, self.references)
        self.assertEqual(len(resegmented_instances), 2)
        self.assertEqual(len(resegmented_instances[0].delays), 11)
        self.assertEqual(len(resegmented_instances[0].elapsed), 11)
        self.assertEqual(len(resegmented_instances[1].delays), 12)
        self.assertEqual(len(resegmented_instances[1].elapsed), 12)
        self.assertEqual(resegmented_instances[0].latency_unit, "word")
        self.assertEqual(resegmented_instances[1].latency_unit, "word")

    def test_segmentlevel_delay_elapsed(self):
        delays_processor = SegmentLevelDelayElapsed(self.instances["ted_1096.wav"])
        # First segment
        stream_delays, stream_elapseds = delays_processor(
            self.resegmented_preds[0], self.references["ted_1096.wav"][0].start_time * 1000)
        # No offset (0.0s) --> same delays
        self.assertEqual(
            stream_delays,
            [2000.0, 2000.0, 3000.0, 3000.0, 3000.0, 4000.0, 4000.0, 4000.0, 4000.0, 4000.0,
             4000.0])
        # No offset (0.0s) --> same first elapsed, and following recomputed elapsed
        self.assertEqual(
            stream_elapseds,
            [3013.3280754089355, 3013.3280754089355, 3467.2186374664307, 3467.2186374664307,
             3467.2186374664307, 4658.461809158325, 4658.461809158325, 4658.461809158325,
             4658.461809158325, 4658.461809158325, 4658.461809158325])

        # Second segment
        stream_delays, stream_elapseds = delays_processor(
            self.resegmented_preds[1], self.references["ted_1096.wav"][1].start_time * 1000)
        # 4.0s offset --> shifted delays by 4000ms
        self.assertEqual(
            stream_delays,
            [0.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 2000.0, 2000.0, 2000.0, 3000.0,
             3000.0, 5340.0])
        # 4.0s offset --> copied elapsed of the first segment, and following recomputed elapsed
        self.assertEqual(
            stream_elapseds,
            [658.4618091583252, 1528.4709930419922, 1528.4709930419922, 1528.4709930419922,
             1528.4709930419922, 1528.4709930419922, 2518.195867538452, 2518.195867538452,
             2518.195867538452, 3472.9058742523193, 3472.9058742523193, 6388.263072967529])

    def test_elapsed_with_offset_at_the_beginning(self):
        stream_instance = {
            "prediction": "also fangen Sie an zu erfinden. Danke. (Applaus)",
            "delays": [313000.0, 313000.0, 313000.0, 314000.0, 314000.0, 314000.0, 323925.375,
                       323925.375],
            "elapsed": [562553.1053543091, 562553.1053543091, 562553.1053543091, 564361.4521026611,
                        564361.4521026611, 564361.4521026611, 582028.8121852875, 582028.8121852875],
        }
        resegmented_preds = ["also fangen Sie an zu erfinden.", "Danke.", "(Applaus)"]

        delays_processor = SegmentLevelDelayElapsed(SimulEvalLogInstance(stream_instance, latency_unit="word"))
        delays_processor.prev_elapsed = 560622.5044727325
        delays_processor.prev_delay = 312000.0
        delays_processor.prev_stream_elapsed = 11121.946849822998

        # Different delay and elapsed in the previous segment, repeated delays and elapsed within
        # the same segment
        stream_delays, stream_elapseds = delays_processor(
            resegmented_preds[0], 311.220000 * 1000)
        self.assertEqual(
            stream_delays,
            [1780.0, 1780.0, 1780.0, 2780.0, 2780.0, 2780.0])
        self.assertEqual(
            stream_elapseds,
            [2710.600881576538, 2710.600881576538, 2710.600881576538, 3588.346748352051,
             3588.346748352051, 3588.346748352051])
        # Different delay and elapsed in the previous segment, single word
        stream_delays, stream_elapseds = delays_processor(
            resegmented_preds[1], 313.780000 * 1000)
        self.assertEqual(
            stream_delays,
            [10145.375])
        self.assertEqual(
            stream_elapseds,
            [17887.360082626343])
        # Same delay and elapsed of the previous segment
        stream_delays, stream_elapseds = delays_processor(
            resegmented_preds[2], 318.730000 * 1000)
        self.assertEqual(
            stream_delays,
            [5195.375])
        self.assertEqual(
            stream_elapseds,
            [12937.360082626343])

    def test_segmentlevel_delay_elapsed_with_negative_delays(self):
        # Create an instance with negative delays in first segment
        stream_instance = {
            "prediction": "fangen Sie an zu erfinden.",
            "delays": [0.0, 1000.0, 3000.0, 3500.0, 4000.0],
            "elapsed": [1000.0, 2000.0, 4000.0, 5000.0, 6000.0],
        }
        resegmented_preds = ["fangen Sie an", "zu erfinden."]

        delays_processor = SegmentLevelDelayElapsed(SimulEvalLogInstance(stream_instance, latency_unit="word"))
        delays_processor.prev_elapsed = None
        delays_processor.prev_delay = None
        delays_processor.prev_stream_elapsed = None

        # First segment with possible negative delays
        stream_delays, stream_elapseds = delays_processor(
            resegmented_preds[0], 1500.0)
        self.assertEqual(stream_delays, [0.0, 0.0, 1500.0])
        self.assertEqual(stream_elapseds, [0.0, 0.0, 1500.0])

        # Second segment without negative delays
        stream_delays, stream_elapseds = delays_processor(
            resegmented_preds[1], 3000.0)
        self.assertEqual(stream_delays, [500.0, 1000.0])
        self.assertEqual(stream_elapseds, [1000.0, 1500.0])


if __name__ == '__main__':
    unittest.main()
