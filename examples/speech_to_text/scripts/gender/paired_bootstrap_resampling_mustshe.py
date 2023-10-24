#!/usr/bin/env python
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
import argparse
from typing import Dict, List, Tuple, Union, Optional
import logging
import numpy as np

from examples.speech_to_text.scripts.gender import mustshe_gender_accuracy as gender_acc
from examples.speech_to_text.scripts.gender.mustshe_gender_accuracy import MuSTSheEntry


def group_by_category(
        available_categories: List[str],
        baseline_stats: List[Dict[str, int]],
        experimental_stats: List[Dict[str, int]],
        mustshe_entries: List[MuSTSheEntry]) -> Dict[str, Dict[str, List]]:
    """
    Returns a dictionary for each category available in the data with
    all the statistics and references.
    """
    stats_by_cat: Dict[str, Dict[str, List]] = {
        c: {"baseline_stats": [], "experimental_stats": [], "mustshe_entries": []} for c in available_categories}
    for baseline_stat, experimental_stat, mustshe_entry in zip(baseline_stats, experimental_stats, mustshe_entries):
        stats_by_cat[mustshe_entry.category]["baseline_stats"].append(baseline_stat)
        stats_by_cat[mustshe_entry.category]["experimental_stats"].append(experimental_stat)
        stats_by_cat[mustshe_entry.category]["mustshe_entries"].append(mustshe_entry)
    if "Global" in available_categories:
        stats_by_cat["Global"] = {
            "baseline_stats": baseline_stats,
            "experimental_stats": experimental_stats,
            "mustshe_entries": mustshe_entries}
    return stats_by_cat


def paired_bootstrap_resample(
        baseline_stats: List[Dict[str, int]],
        experimental_stats: List[Dict[str, int]],
        tsv_reference: List[gender_acc.MuSTSheEntry],
        num_samples: int,
        sample_size: int,
        seed: Optional[int] = None) -> Tuple[int, int, int, int]:
    """
    Returns the statistics of the comparison between the baseline and the experimental model
    for a single category.
    """
    num_sentences = len(baseline_stats)
    if sample_size is None:
        # Defaults to sampling new corpora of the same size as the original.
        # This is not identical to the original corpus since we are sampling with replacement.
        sample_size = num_sentences

    np.random.seed(seed)
    indices_group = np.random.randint(
        low=0, high=num_sentences, size=(num_samples, sample_size))
    baseline_better_cov = 0
    experimental_better_cov = 0
    baseline_better_acc = 0
    experimental_better_acc = 0
    for indices in indices_group:
        sample_baseline_stats: List[Dict[str, int]] = [baseline_stats[i] for i in indices]
        sample_experimental_stats: List[Dict[str, int]] = [experimental_stats[i] for i in indices]
        sample_tsv_reference = [tsv_reference[i] for i in indices]

        baseline_sample = gender_acc.global_scores(sample_baseline_stats, sample_tsv_reference)
        experimental_sample = gender_acc.global_scores(sample_experimental_stats, sample_tsv_reference)

        # Global refers to the statistics of the entire set
        baseline_cov = baseline_sample["Global"].term_coverage
        experimental_cov = experimental_sample["Global"].term_coverage
        baseline_acc = baseline_sample["Global"].gender_accuracy
        experimental_acc = experimental_sample["Global"].gender_accuracy

        baseline_better_cov += int(baseline_cov > experimental_cov)
        experimental_better_cov += int(experimental_cov > baseline_cov)
        baseline_better_acc += int(baseline_acc > experimental_acc)
        experimental_better_acc += int(experimental_acc > baseline_acc)

    return baseline_better_cov, experimental_better_cov, baseline_better_acc, experimental_better_acc


def _calculate_p(
        better_count: int,
        num_samples: int,
        sign_level: float) -> Tuple[float, str]:
    """
    Calculates the p value and returns an asterisk for the print function
    it is greater or less than the significance level.
    """
    p = better_count / num_samples
    if p > 1 - sign_level:
        ast = "*"
    else:
        ast = ""
    return p, ast


def do_print(
        baseline_name: str,
        experimental_name: str,
        results: Dict[str, Dict[str, Union[float, int]]],
        num_samples: int,
        sign_level: float) -> None:
    """
    Executes test, gathers statistics and print them together with the p-value in table format.
    """
    categories = list(results.keys())
    assert len(categories) > 0, "There are no valid categories"
    categories.sort()

    headers = ["SYSTEM", "CATEGORY", "TERM COVERAGE", "GENDER ACCURACY"]
    max_len_sys_name = max(len(baseline_name), len(experimental_name))
    # Truncating too long system names
    len_sys = max_len_sys_name if max_len_sys_name <= 27 else 27
    col_widths = [len_sys + 3, 10, 15, 15]
    header_row = "|".join([header.ljust(width) for header, width in zip(headers, col_widths)])
    separator = "+".join(["-" * width for width in col_widths])
    print(separator)
    print(header_row)
    print(separator)

    for c in categories:
        # H0: experimental better than baseline in coverage
        p_cov_better, ast_cov_better = _calculate_p(
            results[c]["experimental_better_cov"], num_samples, sign_level)
        # H0: baseline better than experimental in coverage
        p_cov_worse, ast_cov_worse = _calculate_p(
            results[c]["baseline_better_cov"], num_samples, sign_level)
        # H0: experimental better than baseline in accuracy
        p_acc_better, ast_acc_better = _calculate_p(
            results[c]["experimental_better_acc"], num_samples, sign_level)
        # H0: baseline better than experimental in accuracy
        p_acc_worse, ast_acc_worse = _calculate_p(
            results[c]["baseline_better_acc"], num_samples, sign_level)
        p_cov, ast_cov = (p_cov_better, ast_cov_better) if p_cov_better > p_cov_worse else (p_cov_worse, ast_cov_worse)
        p_acc, ast_acc = (p_acc_better, ast_acc_better) if p_acc_better > p_acc_worse else (p_acc_worse, ast_acc_worse)

        row_str = "|".join([str(cell).ljust(width) for cell, width in zip([
            baseline_name[:28],
            c,
            f"{results[c]['baseline_all_cov'] * 100:.2f}",
            f"{results[c]['baseline_all_acc'] * 100:.2f}"], col_widths)])
        print(row_str)
        row_str = "|".join([str(cell).ljust(width) for cell, width in zip([
            experimental_name[:28],
            c,
            f"{results[c]['experimental_all_cov'] * 100:.2f} p={p_cov}{ast_cov}",
            f"{results[c]['experimental_all_acc'] * 100:.2f} p={p_acc}{ast_acc}"], col_widths)])
        print(row_str)
        print(separator)


def main(args):
    mustshe_entries = gender_acc.read_mustshe(args.reference_file)
    baseline_translations = gender_acc.read_predictions(args.baseline_file)
    experimental_translations = gender_acc.read_predictions(args.experimental_file)

    # Sanity checks
    assert len(baseline_translations) == len(mustshe_entries), (
        f"Baseline translations have {len(baseline_translations)} lines "
        f"while the reference file has {len(mustshe_entries)} lines.")
    assert len(experimental_translations) == len(mustshe_entries), (
        f"Experimental translations {len(experimental_translations)} lines"
        f"while the reference file has {len(mustshe_entries)} lines.")

    baseline_stats = gender_acc.sentence_level_statistics(baseline_translations, mustshe_entries)
    experimental_stats = gender_acc.sentence_level_statistics(experimental_translations, mustshe_entries)

    baseline_all = gender_acc.global_scores(baseline_stats, mustshe_entries)
    experimental_all = gender_acc.global_scores(experimental_stats, mustshe_entries)

    available_categories = list(baseline_all.keys())
    stats_by_cat = group_by_category(available_categories, baseline_stats, experimental_stats, mustshe_entries)

    results: Dict[str, Dict] = {}
    for cat in args.categories:
        if cat in available_categories:
            cat_results: Dict[str, Union[float, int]] = {
                "baseline_all_cov": baseline_all[cat].term_coverage,
                "baseline_all_acc": baseline_all[cat].gender_accuracy,
                "experimental_all_cov": experimental_all[cat].term_coverage,
                "experimental_all_acc": experimental_all[cat].gender_accuracy}

            base_better_cov, exp_better_cov, base_better_acc, exp_better_acc, = paired_bootstrap_resample(
                baseline_stats=stats_by_cat[cat]["baseline_stats"],
                experimental_stats=stats_by_cat[cat]["experimental_stats"],
                tsv_reference=stats_by_cat[cat]["mustshe_entries"],
                num_samples=args.num_samples,
                sample_size=args.sample_size,
                seed=getattr(args, 'seed'))

            cat_results["experimental_better_cov"] = exp_better_cov
            cat_results["experimental_better_acc"] = exp_better_acc
            cat_results["baseline_better_cov"] = base_better_cov
            cat_results["baseline_better_acc"] = base_better_acc

            results[cat] = cat_results
        else:
            logging.warning(f"Category {cat} is not available in your data")

    do_print(
        baseline_name=os.path.basename(args.baseline_file),
        experimental_name=os.path.basename(args.experimental_file),
        results=results,
        num_samples=args.num_samples,
        sign_level=args.significance_level)


if __name__ == "__main__":
    """
    Script to calculate paired bootstrap resampling for MuST-SHE score metrics, namely
    term_coverage and gender_accuracy (Gaido et al. 2020, https://aclanthology.org/2020.coling-main.350/)
    based on the statistical significance tests for machine translation evaluation.
    If using, please consider citing:
    - D. Fucci et al., "Integrating Language Models into Direct Speech Translation:
      An Inference-Time Solution to Control Gender Inflection", EMNLP 2023.
    
    Version 1.0
    """
    print("paired_bootstrap_resample_mustshe v1.0")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference-file",
        type=str,
        required=True,
        help="Text file containing reference sentences.")
    parser.add_argument(
        "--baseline-file",
        type=str,
        required=True,
        help="Text file containing sentences translated by baseline system. "
             "Sentences should be tokenized at word level")
    parser.add_argument(
        "--experimental-file",
        type=str,
        required=True,
        help="Text file containing sentences translated by the experimental system. "
             "Sentences should be tokenized at word level")
    parser.add_argument(
        "--categories",
        type=str,
        nargs='+',
        required=False,
        choices=["1M", "1F", "2M", "2F", "Global"],
        default=["1M", "1F", "2M", "2F", "Global"],
        help="Categories to evaluate. By default, all categories will be evaluated altogether")
    parser.add_argument(
        "--num-samples",
        type=int,
        required=False,
        default=1000,
        help="Number of comparisons to be executed.")
    parser.add_argument(
        "--sample-size",
        type=int,
        required=False,
        help="Number of sentences sampled for each comparison.")
    parser.add_argument(
        "--significance-level",
        type=float,
        required=False,
        default=0.05,
        help="confidence threshold of obtaining a test statistic under the alternative hypothesis "
             "of no difference between the two paired groups.")
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        help="Seed for random samples.")
    main(parser.parse_args())
