#!/usr/bin/env python3 -u
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

import argparse
import logging
import math
import os
from typing import List, Dict, Set, Tuple
import numpy as np
import pandas as pd

from examples.speech_to_text.xai_metrics.mustshe_scorers import build_scorer
from examples.speech_to_text.xai_metrics.auc_score \
    import check_steps, load_tsv, read_hypos, compute_score, compute_auc, save_plot


_VERSION = "1.0"
LOGGER = logging.getLogger(__name__)


def filter_samples(
        lines: List[str],
        mustshe_reference: List[Dict],
        lower_bound: int,
        upper_bound: int) -> Tuple[Dict[int, List[str]], Dict[int, List[Dict]]]:
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
            n_frames = int(mustshe_reference[sent_id]["n_frames"])
            if lower_bound < n_frames <= upper_bound:
                if step not in step_to_sent:
                    step_to_sent[step] = {}
                    filtered_refs[step] = {}
                step_to_sent[step][sent_id] = line_components[2].strip()
                filtered_refs[step][sent_id] = mustshe_reference[sent_id]
    
    # Order lines following the reference file and turn the dictionaries of dictionaries into
    # dictionaries of lists.
    for step in step_to_sent:
        step_to_sent[step] = [step_to_sent[step][sent_id] for sent_id in sorted(step_to_sent[step].keys())]
        filtered_refs[step] = [filtered_refs[step][sent_id] for sent_id in sorted(filtered_refs[step].keys())]
        
    return step_to_sent, filtered_refs


def merge_dataframes(df_pos: pd.DataFrame, df_hypos: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the dataframe with the hypotheses and the dataframe with the POS information.
    """
    df_pos['id'] = df_pos['ID'] + '_0'
    df = df_hypos.merge(df_pos[['id', 'GENDERTERMS', 'POS']], on='id')
    return df


def find_gt_number(row: pd.Series) -> int:
    """
    Find which gender term in each sentence is being explained.
    Even though the original MuST-SHE presents multiple gender terms in each sentence,
    in this case data should be organized to contain one gender term per row.
    """
    return row['GENDERTERMS'].split(";").index(row['found_term_pairs'])


def find_gt_pos(row: pd.Series) -> str:
    """
    Use the number of the gender term being explained to find its part-of-speech in that sentence's list.
    """
    return row['POS'].split(";")[row['gt_number']]


def select_hypos_by_id(
        steps_to_hypos: Dict[int, List[str]],
        steps_to_refs: Dict[int, List[Dict]],
        ids_to_keep: Set[int]) -> Tuple[Dict[int, List[str]], Dict[int, List[Dict]]]:
    """
    Keep only the hypotheses and refernces with the given IDs.
    """
    for step, refs in steps_to_refs.items():
        steps_to_refs[step] = [refs[i] for i in ids_to_keep]
    for step, hypos in steps_to_hypos.items():
        steps_to_hypos[step] = [hypos[i] for i in ids_to_keep]
    return steps_to_hypos, steps_to_refs


def filter_articles(
        steps_to_hypos: Dict[int, List[str]],
        steps_to_refs: Dict[int, List[Dict]],
        pos_tsv: List[Dict]) -> Tuple[Dict[int, List[str]], Dict[int, List[str]]]:
    """
    Filter out hypotheses and references where the gender term is an article.
    """
    df_hypos = pd.DataFrame(steps_to_refs[0])
    df_pos = pd.DataFrame(pos_tsv)
    df = merge_dataframes(df_pos, df_hypos)
    df['gt_number'] = df.apply(find_gt_number, axis=1)
    df['gt_pos'] = df.apply(find_gt_pos, axis=1)
    ids_to_keep = set(df[df['gt_pos'] != 'Art/Prep'].index)
    steps_to_hypos, steps_to_refs = select_hypos_by_id(steps_to_hypos, steps_to_refs, ids_to_keep)
    return steps_to_hypos, steps_to_refs


def compute_score(
        scorer_name: str,
        steps_to_hypos: Dict[int, List[str]],
        steps_to_refs: Dict[int, List[Dict]],
        categories: List[str] = ["Global"]) -> np.array:
    """
    Compute scores for all the steps in a single output file.
    """
    scores = []
    for step in range(len(steps_to_hypos)):
        assert len(steps_to_hypos[step]) == len(steps_to_refs[step]), \
            f"Different number of hypotheses and references for step {step}"
        scorer = build_scorer(scorer_name)
        for r, h in zip(steps_to_refs[step], steps_to_hypos[step]):
            scorer.add_string(r, h)
        scores.append(scorer.score(categories=categories))
    return scores


def main(args):
    print(f"Version {_VERSION} of FBK AUC MuST-SHE Scorer.")

    # get percentage values
    num_intervals = (args.max_percent // args.perc_step) + 1
    percentage_values = np.array([x * args.perc_step for x in range(num_intervals)])

    # get references
    mustshe_reference = load_tsv(args.tsv_path)

    # get mapping between hypotheses and percentage steps
    hypos_dict = read_hypos(args.output_path)
    all_scores = []
    expected_steps = set(range(num_intervals))
    for file_path, text in hypos_dict:
        step_to_hypos, step_to_refs = filter_samples(text, mustshe_reference, args.lower_bound, args.upper_bound)
        check_steps(file_path, step_to_hypos, expected_steps)  # check if there are missing or extra steps

        # filter out articles if needed
        if args.no_articles:
            assert args.mustshe_pos and os.path.isfile(args.mustshe_pos), \
                "The MuST-SHE POS extension file is needed to filter out articles."
            pos_tsv = load_tsv(args.mustshe_pos)
            step_to_hypos, step_to_refs = filter_articles(step_to_hypos, step_to_refs, pos_tsv)

        # compute scores
        scores = compute_score(args.scorer, step_to_hypos, step_to_refs, categories=args.categories)
        all_scores.append(scores)

    scores_array = np.array(all_scores)
    # average scores across all files
    scores = np.mean(scores_array, axis=0)

    # print/plot AUC
    if args.fig_path is not None:
        save_plot(percentage_values, scores, args.fig_path, args.scorer)
    auc_score = compute_auc(scores, percentage_values)
    print(f"AUC ({args.scorer} {'without articles' if args.no_articles else 'with articles'}): {auc_score:.3f}")

    # save data
    if args.data_path is not None:
        np.save(os.path.join(args.data_path, "percentage.npy"), percentage_values)
        np.save(os.path.join(args.data_path, "scores.npy"), scores)


if __name__ == "__main__":
    """
    This script computes the AUC score from predictions generated for the insertion/deletion metric 
    for the MuST-SHE dataset.
    """
    parser = argparse.ArgumentParser(
        description="Calculate (and plot) AUC with CI from predictions generated "
                    "for insertion/deletion metric.")
    parser.add_argument(
        "--tsv-path",
        required=True,
        type=str,
        help="Path to the TSV file containing the hypotheses which were explained.")
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
        "--max-percent",
        type=int,
        default=100,
        help="Maximum percentage of input features to insert/delete.")
    parser.add_argument(
        "--scorer",
        type=str,
        default="gender_accuracy",
        choices=["gender_accuracy", "gender_coverage", "gender_flip_rate"],
        help="Metric for the evaluation of gender explanations.")
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=["Global"],
        choices=["Global", "1F", "1M", "2F", "2M", "3F", "3M", "4F", "4M"],
        help="Categories of the MuST-SHE dataset to consider.")
    parser.add_argument(
        "--no-articles", action="store_true", help="Consider only gender terms which are not articles.")
    parser.add_argument(
        "--mustshe-pos",
        type=str,
        default=None,
        help="Path to the MuST-SHE extension file containing POS information.")
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
