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
import argparse
import os
import re
from typing import List, NamedTuple

import numpy as np

_SUPPORTED_METRICS = ['cps', 'cpl', 'lpb']
_VERSION = "1.0"
_CITATION = r"""@misc{papi2022direct,
      title={{Direct Speech Translation for Automatic Subtitling}}, 
      author={Sara Papi and Marco Gaido and Alina Karakanta and Mauro Cettolo and Matteo Negri and Marco Turchi},
      year={2022}
}"""


try:
    import srt
except ImportError:
    print("Please install the srt package with 'pip install srt' and try again.")
    exit(1)


class ComplianceMetric(NamedTuple):
    name: str
    upperbound: float
    mean: float
    stdev: float
    maximum: float
    total: int
    num_compliant: int

    @property
    def score(self):
        return float(self.num_compliant) / self.total

    @staticmethod
    def _format_number(num: float, precision: int):
        return '{:.{}f}'.format(num, precision)

    def score_string(self, precision: int):
        return f"{self.name.upper()}: {self._format_number(self.score * 100, precision)}%"

    def json_string(self, precision: int):
        return os.linesep.join([
            '{',
            f' "metric": "{self.name.upper()} <= {self.upperbound}",',
            f' "score": "{self._format_number(self.score * 100, precision)}%",',
            f' "mean": {self._format_number(self.mean, precision)},',
            f' "stdev": {self._format_number(self.stdev, precision)},',
            f' "total": {self._format_number(self.total, precision)},',
            f' "compliant": {self._format_number(self.num_compliant, precision)},',
            f' "version": "{_VERSION}"',
            '}'])


class SubtitleComplianceStats:
    """
    Helper class which computes, holds, and formats the compliance metrics.
    """
    def __init__(self, cps: List[float], cpl: List[float], lpb: List[int]):
        self.cps = cps
        self.cpl = cpl
        self.lpb = lpb

    @staticmethod
    def clean_content(subtitle, remove_parenthesis_content=False):
        """
        Takes the subtitling content and removes content in parentheses (either round or square)
        if `remove_parenthesis_content` is set.

        >>> from datetime import timedelta
        >>> sub = srt.Subtitle(
        ...    1, timedelta(0), timedelta(0),
        ...    'A cool line (Laughter!)\\nwith several content [Noise]')
        >>> SubtitleComplianceStats.clean_content(sub)
        'A cool line (Laughter!)\\nwith several content [Noise]'
        >>> SubtitleComplianceStats.clean_content(sub, remove_parenthesis_content=True)
        'A cool line \\nwith several content '
        """
        if remove_parenthesis_content:
            return re.sub('\(.*?\)|\[.*?\]', '', subtitle.content)
        return subtitle.content

    @classmethod
    def from_subtitles(
            cls, subtitles: List[srt.Subtitle], remove_parenthesis_content: bool = False):
        """
        Creates the class from a subtitle content.
        """
        cps = []
        cpl = []
        lpb = []
        for sub in subtitles:
            content = cls.clean_content(sub, remove_parenthesis_content)
            block_len = 0
            num_lines = 0
            for ln in content.split('\n'):
                line_len = len(ln.strip())
                block_len += line_len
                num_lines += 1
                cpl.append(line_len)
            lpb.append(num_lines)
            cps.append(block_len / (sub.end - sub.start).total_seconds())
        return cls(cps, cpl, lpb)

    @classmethod
    def merge(cls, subtitle_stats: List):
        cps = []
        cpl = []
        lpb = []
        for stats in subtitle_stats:
            cps.extend(stats.cps)
            cpl.extend(stats.cpl)
            lpb.extend(stats.lpb)
        return cls(cps, cpl, lpb)

    def metric(self, name: str, upperbound: float) -> ComplianceMetric:
        assert name in _SUPPORTED_METRICS, f"Unsupported metric '{name}'"
        stats = np.array(getattr(self, name))
        return ComplianceMetric(
            name,
            upperbound,
            np.mean(stats),
            np.std(stats),
            np.max(stats),
            len(stats),
            np.sum(stats <= upperbound))

    def report(self, metric: str, upperbound: float, precision: int, quiet: bool) -> str:
        compliance_metric = self.metric(metric, upperbound)
        if quiet:
            return compliance_metric.score_string(precision)
        else:
            return compliance_metric.json_string(precision)


def main(args):
    """
    Computes the:
    - character per second (CPS) metric for each block;
    - character per line (CPL) metric for each line;
    - line per block (LPB) metric for each block;
    for all the SRT files specified.
    It prints the percentage of compliance with respect to the given maximum CPS/CPL/LPB allowed,
    as well as  statistics about the CPS/CPL/LPB.

    The metrics are computed in the same way as done in `"Direct Speech Translation for Automatic
    Subtitling" (Papi et al., 2022) <https://arxiv.org/pdf/2209.13192.pdf>`_.

    If using this script, please consider citing it.
    """
    print(f"Version {_VERSION} of FBK Subtitle Compliance tool.")
    if args.cite:
        print("BibTeX for metrics:\n" + _CITATION)
        exit()
    all_stats = []
    for srt_file in args.srt_file:
        with open(srt_file) as f:
            subtitles = list(srt.parse(f))

        subtitle_stats = SubtitleComplianceStats.from_subtitles(
            subtitles, args.remove_parenthesis_content)
        all_stats.append(subtitle_stats)
        if not args.quiet and len(args.srt_file) > 1:
            print(f"Compliance metrics for {srt_file}")
            for m in args.metrics:
                print(subtitle_stats.report(
                    m, getattr(args, f'max_{m}'), args.width, args.quiet))
    if not args.quiet:
        print(f"Overall compliance metrics")

    overall_subtitle_stats = SubtitleComplianceStats.merge(all_stats)
    for m in args.metrics:
        print(overall_subtitle_stats.report(
            m, getattr(args, f'max_{m}'), args.width, args.quiet))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # I/O related arguments
    parser.add_argument(
        '--srt-file', type=str, nargs='+',
        help="the SRT file(s) for which the metrics should be computed")

    # Metric selection
    parser.add_argument(
        '--metrics', '-m', choices=_SUPPORTED_METRICS, nargs='+', default=['cps', 'cpl', 'lpb'],
        help='Space-delimited list of metrics to compute.')

    # CPS-related arguments
    parser.add_argument(
        '--max-cps', type=float, default=21,
        help="maximum value allowed for the CPS (characters per second) "
             "to be compliant with guidelines.")

    # CPL-related arguments
    parser.add_argument(
        '--max-cpl', type=int, default=42,
        help="maximum value allowed for the CPL (characters per line) "
             "to be compliant with guidelines.")

    # LPB-related arguments
    parser.add_argument(
        '--max-lpb', type=int, default=2,
        help="maximum value allowed for the LPB (line per block) "
             "to be compliant with guidelines.")

    # Reporting related arguments
    parser.add_argument(
        '--quiet', '-q', default=False, action='store_true',
        help='print only the computed score.')
    parser.add_argument(
        '--width', '-w', type=int, default=1,
        help='floating point width.')

    # Text preprocessing:
    parser.add_argument(
        "--remove-parenthesis-content", default=False, action='store_true',
        help="if set, content in parenthesis is removed before computing the score.")

    parser.add_argument(
        '--cite', default=False, action='store_true',
        help='print the bibtex for citation and exit.')

    parsed_args = parser.parse_args()
    srt_files_specified = \
        getattr(parsed_args, 'srt_file') is not None and len(parsed_args.srt_file) > 0
    if not (srt_files_specified or parsed_args.cite):
        print("--srt-file is required")
        parser.print_usage()
        exit(1)
    main(parsed_args)
