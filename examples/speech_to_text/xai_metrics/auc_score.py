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
import csv
import glob
import logging
import math
import os
from typing import List, Dict, Tuple
import numpy as np

from fairseq.scoring import build_scorer


_VERSION = "1.0"
LOGGER = logging.getLogger(__name__)


def filter_samples(
        lines: List[str],
        tsv: List[Dict],
        lower_bound: int,
        upper_bound: int) -> Tuple[Dict[int, List[str]], Dict[int, List[str]]]:
    """
    Filter hypotheses in a text based on percentage steps, and hypotheses and
    references based on the number of frames.
    Returns two dictionaries that map each step to corresponding hypotheses/reference.
    """
    step_to_sent = {}
    filtered_refs = {}
    for l in lines:
        if l.startswith("D-"):
            line_components = l.split("\t")
            _, sent_id, step = line_components[0].split("-")
            step = int(step)
            sent_id = int(sent_id)
            n_frames = int(tsv[sent_id]["n_frames"])
            if lower_bound < n_frames <= upper_bound:
                if step not in step_to_sent:
                    step_to_sent[step] = []
                    filtered_refs[step] = []
                step_to_sent[step].append(line_components[2].strip())
                filtered_refs[step].append(tsv[sent_id]["tgt_text"])
    return step_to_sent, filtered_refs


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


def load_tsv(file_path: str) -> List[Dict]:
    """
    Load tsv file and returns a list of dictionaries, one per sample.
    """
    data = []
    with open(file_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            data.append(row)
    return data


def compute_score(
        scorer_name: str,
        steps_to_refs: Dict[int, List[str]],
        steps_to_hypos: Dict[int, List[str]]) -> List[float]:
    """
    Compute scores for a single file for all the steps.
    """
    scores = []
    for step in range(len(steps_to_hypos)):
        assert len(steps_to_hypos[step]) == len(steps_to_refs[step]), \
            f"Different number of hypotheses and references for step {step}"
        scorer = build_scorer(scorer_name, tgt_dict=None)
        for r, h in zip(steps_to_refs[step], steps_to_hypos[step]):
            scorer.add_string(r, h)
        scores.append(scorer.score())
    return scores


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


def check_steps(
        file_path: str,
        step_to_hypos: Dict[int, List[str]],
        expected_steps: set) -> None:
    """
    Check if there are missing or extra steps in `step_to_hypos`.
    """
    actual_steps = set(step_to_hypos.keys())

    missing_steps = expected_steps - actual_steps
    extra_steps = actual_steps - expected_steps

    if missing_steps:
        missing_steps_str = ', '.join(map(str, sorted(missing_steps)))
        raise ValueError(f"Missing steps for file {file_path}: {missing_steps_str}")
    if extra_steps:
        extra_steps_str = ', '.join(map(str, sorted(extra_steps)))
        LOGGER.warning(f"Extra steps found for file {file_path}: {extra_steps_str}")


def main(args):
    print(f"Version {_VERSION} of FBK AUC Scorer.")

    # get percentage values
    num_intervals = (100 // args.perc_step) + 1
    percentage_values = np.array([x * args.perc_step for x in range(num_intervals)])

    # get mapping between hypotheses/references and percentage steps and compute scores
    tsv = load_tsv(args.tsv_path)
    hypos_dict = read_hypos(args.output_path)
    all_scores = []
    expected_steps = set(range(num_intervals))
    for file_path, text in hypos_dict:
        step_to_hypos, step_to_refs = filter_samples(text, tsv, args.lower_bound, args.upper_bound)
        check_steps(file_path, step_to_hypos, expected_steps)  # check if there are missing or extra steps
        scores = compute_score(args.scorer, step_to_refs, step_to_hypos)
        all_scores.append(scores)

    scores_array = np.array(all_scores)
    scores = np.mean(scores_array, axis=0)

    # print/plot AUC
    if args.fig_path is not None:
        save_plot(percentage_values, scores, args.fig_path, args.scorer)

    auc_score = compute_auc(scores, percentage_values)
    print(f"AUC: {auc_score}")

    # save data
    if args.data_path is not None:
        np.save(os.path.join(args.data_path, "percentage.npy"), percentage_values)
        np.save(os.path.join(args.data_path, "scores.npy"), scores)

    # plot AUC
    if args.fig_path is not None:
        save_plot(percentage_values, scores, args.fig_path, args.scorer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate (and plot) AUC with CI from predictions generated "
                    "for insertion/deletion metric.")
    parser.add_argument(
        "--output-path",
        help="Path to a single output file or to the folder containing multiple output files. "
             "If the latter, the scores of different output files are averaged.")
    parser.add_argument(
        "--tsv-path",
        required=True,
        type=str,
        help="Path to the tsv file of the original samples, where the references and the number "
             "of frames for each sample are stored.")
    parser.add_argument(
        "--perc-step",
        type=int,
        help="Percentage step size according to which "
             "computing insertion/deletion of input features.")
    parser.add_argument(
        "--scorer", type=str, default="wer_max", help="Metric for the evaluation of the task.")
    parser.add_argument(
        "--lower-bound",
        type=int,
        default=-math.inf,
        help="Lower bound of the number of frames for which to compute the metric. If not specified, "
             "there will be no lower bound.")
    parser.add_argument(
        "--upper-bound",
        type=int,
        default=+math.inf,
        help="Upper bound of the number of frames for which to compute the metric. If not specified, "
             "there will be no upper bound.")
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to the folder where npy arrays containing the data to plot will be saved.")
    parser.add_argument(
        "--fig-path", type=str, default=None, help="Path to the file where the plot is saved.")
    args = parser.parse_args()
    main(args)
