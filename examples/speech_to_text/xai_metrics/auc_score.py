#!/usr/bin/env python3 -u
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
import glob
import logging
import os
from typing import List, Dict, Tuple
import numpy as np

from fairseq.scoring import build_scorer


_VERSION = "1.0"
LOGGER = logging.getLogger(__name__)


def filter_hypos(
        lines: List[str], num_intervals: int, ref_length: int, file_path: str
) -> Dict[int, List[str]]:
    """
    Filter hypotheses in a text based on percentage steps.
    Returns a dictionary which maps each step to corresponding hypotheses.
    """
    step_to_sent = {}
    for l in lines:
        if l.startswith("D-"):
            line_components = l.split("\t")
            _, sent_id, step = line_components[0].split("-")
            step = int(step)
            sent_id = int(sent_id)
            if step not in step_to_sent:
                step_to_sent[step] = {}
            step_to_sent[step][sent_id] = line_components[2].strip()

    for step in range(num_intervals):
        if step not in step_to_sent:
            raise KeyError(f"Step {step} not available in file {file_path}.")
        step_text = step_to_sent[step]

        ordered_lines = []
        for i in range(ref_length):
            if i not in step_text:
                raise KeyError(f"Sentence {i} for step {step} not available in file {file_path}.")
            ordered_lines.append(step_text[i])
        step_to_sent[step] = ordered_lines

    return step_to_sent


def read_hypos(hypo_path: str) -> List[Tuple[str, List[str]]]:
    """
    If hypo_path is a file, it reads the file and returns its content.
    Otherwise, hypo_path is considered as prefix and returns the content
    of all the files that match the prefix.
    """
    if os.path.isfile(hypo_path):
        LOGGER.info(f"Reading {hypo_path}")
        with open(hypo_path, 'r') as f:
            return [(hypo_path, f.readlines())]
    else:
        file_paths = glob.glob(hypo_path + "/*")
        if len(file_paths) == 0:
            raise ValueError(f"Invalid path: {hypo_path}")
        file_dict = []
        for file_path in glob.glob(hypo_path + "/*"):
            LOGGER.info(f"Reading {file_path}")
            if os.path.isfile(file_path):
                with open(file_path, 'r') as f:
                    file_dict.append((file_path, f.readlines()))
        return file_dict


def compute_single_score(scorer: str, refs: List[str], hypos: List[str]) -> float:
    """
    Compute score for a single output file in a single step.
    """
    assert len(refs) == len(hypos)
    scorer = build_scorer(scorer, tgt_dict=None)
    for r, h in zip(refs, hypos):
        scorer.add_string(r, h)
    return scorer.score()


def compute_score(
        scorer: str,
        refs: List[str],
        hypos_list: List[Dict[int, List[str]]]) -> np.array:
    """
    Compute scores for all the available output files and for all the steps.
    Returns the averaged scores across the various output files.
    """
    all_scores = []
    for hypos_dict in hypos_list:
        scores = []
        for step in hypos_dict:
            scores.append(compute_single_score(scorer, refs, hypos_dict[step]))
        all_scores.append(scores)
    scores_array = np.array(all_scores)
    return np.mean(scores_array, axis=0)


def compute_auc(scores: np.array, percentage_values: np.array) -> float:
    """
    Returns the AUC.
    """
    return np.trapz(scores, percentage_values)


def save_plot(percentages: np.array, scores: np.array, save_path: str, scorer: str) -> None:
    if "wer" in scorer:
        _scorer = "WER"
    elif "bleu" in scorer:
        _scorer = "BLEU"
    else:
        _scorer = ""
    import matplotlib.pyplot as plt
    plt.plot(percentages, scores, linestyle='-')
    # highlight the area under the curve
    plt.fill_between(percentages, scores, 0, color='skyblue', alpha=0.3)
    plt.xlabel('Step (%)')
    plt.ylabel(f'{_scorer}')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(save_path)


def main(args):
    print(f"Version {_VERSION} of FBK AUC Scorer.")

    assert 0 <= args.conf_interval <= 100, "--conf-interval must be in the range [0-100]."

    # get percentage values
    num_intervals = (100 // args.perc_step) + 1
    percentage_values = np.array([(x + 1) * args.perc_step for x in range(num_intervals)])

    # get references
    with open(args.reference, 'r') as f:
        refs = f.readlines()

    # get mapping between hypotheses and percentage steps
    hypos_dict = read_hypos(args.output_path)
    hypos_mapping_list = []
    for file_path, text in hypos_dict:
        step_to_sent = filter_hypos(text, num_intervals, len(refs), file_path)
        hypos_mapping_list.append(step_to_sent)

    # compute scores
    scores = compute_score(args.scorer, refs, hypos_mapping_list)

    # print/plot AUC
    if args.fig_path is not None:
        save_plot(percentage_values, scores, args.fig_path, args.scorer)
    auc_score = compute_auc(scores, percentage_values)
    print(f"AUC: {auc_score}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate (and plot) AUC with CI from predictions generated "
                    "for insertion/deletion metric.")
    parser.add_argument("--reference", type=str, help="Path to the reference file.")
    parser.add_argument(
        "--output-path",
        help="Path to a single output file or to the folder containing multiple output files. "
             "If the latter, the scores of different output files are averaged.")
    parser.add_argument(
        "--perc-step",
        type=int,
        help="Percentage step size according to which "
             "computing insertion/deletion of input features.")
    parser.add_argument(
        "--scorer", type=str, default="wer_max", help="Metric for the evaluation of the task.")
    parser.add_argument(
        "--fig-path", type=str, default=None, help="Path to the file where the plot is saved.")
    args = parser.parse_args()
    main(args)
