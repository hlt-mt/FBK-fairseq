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

import srt

from examples.speech_to_text.scripts import subtitle_compliance
from examples.speech_to_text.scripts.subtitle_compliance import SubtitleComplianceStats


class TestSubtitleCompliance(unittest.TestCase):
    def test_doctest(self):
        import doctest
        results = doctest.testmod(m=subtitle_compliance)
        self.assertEqual(0, results.failed)

    @staticmethod
    def get_example_content(name):
        path = os.path.join(os.path.dirname(__file__), f"resources/{name}")
        with open(path) as f:
            return srt.parse(f.read())

    def test_basic(self):
        subtitles = self.get_example_content("sample_de_01.srt")
        stats = SubtitleComplianceStats.from_subtitles(subtitles)
        cps_metric = stats.metric("cps", 21)
        self.assertEqual(cps_metric.total, 4)
        self.assertEqual(cps_metric.num_compliant, 4)
        self.assertEqual(cps_metric.name, "cps")
        self.assertEqual(cps_metric.upperbound, 21)
        self.assertEqual(cps_metric.maximum, 18.650793650793652)
        self.assertEqual(cps_metric.mean, 16.722987261780368)
        self.assertEqual(cps_metric.stdev, 1.844207837519194)
        self.assertEqual(cps_metric.score, 1.0)
        cps_metric2 = stats.metric("cps", 16.6)
        self.assertEqual(cps_metric2.total, 4)
        self.assertEqual(cps_metric2.num_compliant, 2)
        cpl_metric = stats.metric("cpl", 21)
        self.assertEqual(cpl_metric.total, 8)
        self.assertEqual(cpl_metric.num_compliant, 1)
        self.assertEqual(cpl_metric.name, "cpl")
        self.assertEqual(cpl_metric.upperbound, 21)
        self.assertEqual(cpl_metric.maximum, 40)
        self.assertEqual(cpl_metric.mean, 32.5)
        self.assertEqual(cpl_metric.stdev, 7.106335201775948)
        self.assertEqual(cpl_metric.score, 0.125)
        lpb_metric = stats.metric("lpb", 2)
        self.assertEqual(lpb_metric.total, 4)
        self.assertEqual(lpb_metric.num_compliant, 4)
        self.assertEqual(lpb_metric.name, "lpb")
        self.assertEqual(lpb_metric.upperbound, 2)
        self.assertEqual(lpb_metric.maximum, 2)
        self.assertEqual(lpb_metric.mean, 2)
        self.assertEqual(lpb_metric.stdev, 0.0)
        self.assertEqual(lpb_metric.score, 1.0)
        self.assertEqual(cpl_metric.json_string(2), """{
 "metric": "CPL <= 21",
 "score": "12.50%",
 "mean": 32.50,
 "stdev": 7.11,
 "total": 8.00,
 "compliant": 1.00,
 "version": "1.1"
}""")
        self.assertEqual(cpl_metric.score_string(2), "CPL: 12.50%")

    def test_merge(self):
        subtitles1 = self.get_example_content("sample_de_01.srt")
        stats1 = SubtitleComplianceStats.from_subtitles(subtitles1)
        subtitles2 = self.get_example_content("sample_de_02.srt")
        stats2 = SubtitleComplianceStats.from_subtitles(subtitles2)
        merged = SubtitleComplianceStats.merge([stats1, stats2])
        cps_metric = merged.metric("cps", 21)
        self.assertEqual(cps_metric.total, 8)
        self.assertEqual(cps_metric.num_compliant, 7)
        self.assertEqual(cps_metric.name, "cps")
        self.assertEqual(cps_metric.score, 0.875)

    def test_brackets(self):
        stats_nobrackets = SubtitleComplianceStats.from_subtitles(
            self.get_example_content("sample_brackets.srt"),
            remove_parenthesis_content=True)
        stats = SubtitleComplianceStats.from_subtitles(
            self.get_example_content("sample_brackets.srt"))
        cpl_nobrackets = stats_nobrackets.metric("cpl", 42)
        cpl = stats.metric("cpl", 42)
        self.assertEqual(cpl_nobrackets.total, 4)
        self.assertEqual(cpl.total, 4)
        self.assertEqual(cpl_nobrackets.num_compliant, 4)
        self.assertEqual(cpl.num_compliant, 3)
        self.assertEqual(cpl_nobrackets.mean, 20.5)
        self.assertEqual(cpl.mean, 24.75)

    def test_confidence_interval(self):
        subtitles = self.get_example_content("sample_de_01.srt")
        stats = SubtitleComplianceStats.from_subtitles(subtitles)
        cpl_metric = stats.metric("cpl", 21, ci=True)
        self.assertEqual(cpl_metric.ci.mu, 0.127)
        self.assertEqual(cpl_metric.ci.var, 0.1875)
        self.assertEqual(cpl_metric.json_string(2), """{
 "metric": "CPL <= 21",
 "score": "12.50% (μ = 12.70 ± 18.75)",
 "mean": 32.50,
 "stdev": 7.11,
 "total": 8.00,
 "compliant": 1.00,
 "version": "1.1"
}""")


if __name__ == '__main__':
    unittest.main()
