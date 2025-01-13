#!/usr/bin/env python3
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
# Credits: this implementation is inspired by the BLEU statistical significance
# implementation in https://github.com/pytorch/translate/.
import argparse
from typing import List, NamedTuple, Optional

import numpy as np
import pandas as pd
import jiwer


def get_sufficient_stats(hypotheses: List[str], references: List[str]) -> pd.DataFrame:
    """
    Computes the sentence-level statistics required to compute WER.
    """
    assert len(hypotheses) == len(references), \
        f"There are {len(hypotheses)} hypothesis sentences but {len(references)} reference " \
        "sentences"
    sufficient_stats: List[List[int]] = []
    for sentence, ref in zip(hypotheses, references):
        sentence_wer_stats = jiwer.process_words(ref, sentence)
        sufficient_stats.append([
            sentence_wer_stats.hits,
            sentence_wer_stats.substitutions,
            sentence_wer_stats.insertions,
            sentence_wer_stats.deletions])
    return pd.DataFrame(
        sufficient_stats, columns=["hits", "substitutions", "insertions", "deletions"])


class PairedBootstrapOutput(NamedTuple):
    baseline_wer: float
    evaluated_wer: float
    num_samples: int
    # Number of samples where the baseline was better than the evaluated system.
    baseline_better: int
    # Number of samples where the baseline and evaluated system had identical WER score.
    num_equal: int
    # Number of samples where the evaluated system was better than baseline.
    evaluated_better: int


def compute_wer_from_stats(stats: pd.DataFrame):
    """
    Returns the WER score starting from sentence-level statistics.
    The formula is taken from jiwer's `process_words` implementation.
    """
    corpus_stats = stats.sum(axis=0)
    edits = float(corpus_stats.substitutions + corpus_stats.deletions + corpus_stats.insertions)
    total = float(corpus_stats.hits + corpus_stats.substitutions + corpus_stats.deletions)
    return edits / total


def paired_bootstrap_resample(
    baseline_stats: pd.DataFrame,
    evaluated_stats: pd.DataFrame,
    num_samples: int = 1000,
    sample_size: Optional[int] = None,
) -> PairedBootstrapOutput:
    """
    From `Statistical significance tests for machine translation evaluation (Koehn, 2004)
    <http://aclweb.org/anthology/W04-3250>`_.
    """
    assert len(baseline_stats) == len(evaluated_stats), \
        f"Length mismatch - baseline has {len(baseline_stats)} lines while evaluated has " \
        f"{len(evaluated_stats)} lines."
    num_sentences = len(baseline_stats)
    if not sample_size:
        # Defaults to sampling new corpora of the same size as the original.
        # This is not identical to the original corpus since we are sampling with replacement.
        sample_size = num_sentences
    indices = np.random.randint(low=0, high=num_sentences, size=(num_samples, sample_size))

    baseline_better: int = 0
    evaluated_better: int = 0
    num_equal: int = 0
    for index in indices:
        baseline_wer = compute_wer_from_stats(baseline_stats.iloc[index])
        evaluated_wer = compute_wer_from_stats(evaluated_stats.iloc[index])
        if evaluated_wer < baseline_wer:
            evaluated_better += 1
        elif baseline_wer < evaluated_wer:
            baseline_better += 1
        else:
            # If the baseline corpus and new corpus are identical, this
            # degenerate case may occur.
            num_equal += 1

    return PairedBootstrapOutput(
        baseline_wer=compute_wer_from_stats(baseline_stats),
        evaluated_wer=compute_wer_from_stats(evaluated_stats),
        num_samples=num_samples,
        baseline_better=baseline_better,
        num_equal=num_equal,
        evaluated_better=evaluated_better)


def paired_bootstrap_resample_from_files(
        reference_file: str,
        baseline_file: str,
        evaluated_file: str,
        num_samples: int = 1000,
        sample_size: Optional[int] = None) -> PairedBootstrapOutput:
    with open(reference_file, "r") as f:
        references: List[str] = [line for line in f]

    with open(baseline_file, "r") as f:
        baseline_hypotheses: List[str] = [line for line in f]
    baseline_stats: pd.DataFrame = get_sufficient_stats(
        hypotheses=baseline_hypotheses, references=references)

    with open(evaluated_file, "r") as f:
        evaluated_hypotheses: List[str] = [line for line in f]
    evaluated_stats: pd.DataFrame = get_sufficient_stats(
        hypotheses=evaluated_hypotheses, references=references)

    return paired_bootstrap_resample(
        baseline_stats=baseline_stats,
        evaluated_stats=evaluated_stats,
        num_samples=num_samples,
        sample_size=sample_size)


def main():
    parser = argparse.ArgumentParser()
    tokenization_warning = "Input should be tokenized and punctuation should be removed.",
    parser.add_argument(
        "--reference-file",
        type=str,
        required=True,
        help=f"Text file containing reference sentences. {tokenization_warning}",
    )
    parser.add_argument(
        "--baseline-file",
        type=str,
        required=True,
        help="Text file containing sentences translated by the baseline system. "
             f" {tokenization_warning}",
    )
    parser.add_argument(
        "--evaluated-file",
        type=str,
        required=True,
        help="Text file containing sentences translated by the evaluated system. "
             f"{tokenization_warning}",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        required=False,
        default=1000,
        help="Number of comparisons to be executed.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        required=False,
        help="Number of sentences sampled for each comparison.",
    )
    args = parser.parse_args()

    output = paired_bootstrap_resample_from_files(
        reference_file=args.reference_file,
        baseline_file=args.baseline_file,
        evaluated_file=args.evaluated_file,
        num_samples=args.num_samples,
        sample_size=getattr(args, "sample_size", None))

    print(f"Baseline system WER: {output.baseline_wer:.4f}")
    print(f"Evaluated system WER: {output.evaluated_wer:.4f}")
    print(f"WER delta: {output.evaluated_wer - output.baseline_wer:.4f} ")
    print(f"Baseline system better confidence: {output.baseline_better / output.num_samples:.2%}")
    print(f"Evaluated system better confidence: {output.evaluated_better / output.num_samples:.2%}")


if __name__ == "__main__":
    main()
