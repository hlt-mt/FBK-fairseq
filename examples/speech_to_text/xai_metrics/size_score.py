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
import os
from typing import Tuple

import numpy as np
import torch
from torch import Tensor

from examples.speech_to_text.occlusion_explanation.normalizers import NORMALIZATION_REGISTRY, get_normalizer
from examples.speech_to_text.occlusion_explanation.utils import read_feature_attribution_maps_from_h5


_VERSION = "1.0"


def get_values_above_threshold(
        fbank_explanation: Tensor,
        tgt_explanation: Tensor,
        threshold: float) -> Tuple[int, int]:
    """
    Returns the number of features of respectively `fbank_explanation` and
    `tgt_explanation` with values higher than `threshold`. In `tgt_explanation`,
    padded values are not considered.
    """
    flat_fbank_explanation = torch.flatten(fbank_explanation)
    upper_triangular_mask = torch.tril(torch.ones_like(tgt_explanation[:, :, 0])).type(torch.bool)
    tgt_explanation = torch.where(upper_triangular_mask, tgt_explanation.squeeze(-1), -float('inf'))
    flat_tgt_explanation = torch.flatten(tgt_explanation)
    return (
        torch.sum(flat_fbank_explanation >= threshold).item(),
        torch.sum(flat_tgt_explanation >= threshold).item())


def count_values(fbank_explanation: Tensor, tgt_explanation: Tensor) -> Tuple[int, int]:
    """
    Returns the number of features in explanations. For target embeddings explanations
    padded values are not considered.
    """
    fbank_count = fbank_explanation.numel()
    upper_triangular_mask = torch.tril(torch.ones_like(tgt_explanation[:, :, 0]))
    tgt_count = torch.sum(upper_triangular_mask).item() * tgt_explanation.size(2)
    return fbank_count, tgt_count


def save_plot(
        thresholds: np.array,
        scores_fbank: np.array,
        scores_tgt: np.array,
        save_path: str) -> None:
    """
    Plots and saves graph of different size scores with different thresholds.
    """
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(thresholds, scores_fbank, linestyle='-', color='blue', label='Scores FBank')
    ax1.fill_between(thresholds, scores_fbank, 0, color='skyblue', alpha=0.3)
    ax1.set_xlabel('Step (%)')
    ax1.set_ylabel('Number of values (%)')
    ax1.grid(False)
    ax2.plot(thresholds, scores_tgt, linestyle='-', color='green', label='Scores TGT')
    ax2.fill_between(thresholds, scores_tgt, 0, color='lightgreen', alpha=0.3)
    ax2.set_xlabel('Step (%)')
    ax2.set_ylabel('Number of values (%)')
    ax2.grid(False)
    plt.tight_layout()
    plt.savefig(save_path)


def _main(args: argparse.Namespace) -> None:
    print(f"Version {_VERSION} of Size score.")

    explanations = read_feature_attribution_maps_from_h5(args.explanation_path)

    normalizers = [get_normalizer(norm) for norm in args.normalizer]

    fbank_results = []
    tgt_results = []
    thresholds = [t / 100 for t in range(args.start_threshold, 100, args.step)] + [1]
    for t in thresholds:
        fbank_sizes = []
        tgt_sizes = []
        for sample in explanations:
            for norm in normalizers:
                norm = norm()
                explanations[sample]["fbank_heatmap"], explanations[sample]["tgt_embed_heatmap"] = norm(
                    explanations[sample]["fbank_heatmap"], explanations[sample]["tgt_embed_heatmap"])

            values_fbank, values_tgt = get_values_above_threshold(
                explanations[sample]["fbank_heatmap"], explanations[sample]["tgt_embed_heatmap"], t)
            fbank_count, tgt_count = count_values(
                explanations[sample]["fbank_heatmap"], explanations[sample]["tgt_embed_heatmap"])

            fbank_sizes.append(values_fbank / fbank_count * 100)
            tgt_sizes.append(values_tgt / tgt_count * 100)
        
        # Average at corpus level for a single step in thresholds
        fbank_results.append(sum(fbank_sizes) / len(fbank_sizes))
        tgt_results.append(sum(tgt_sizes) / len(tgt_sizes))

    print(f"AUC fbanks: {np.trapz(fbank_results, thresholds)}")
    print(f"AUC tgt: {np.trapz(tgt_results, thresholds)}")

    # save data
    if args.data_path is not None:
        np.save(os.path.join(args.data_path, "percentage.npy"), thresholds)
        np.save(os.path.join(args.data_path, "fbank_scores.npy"), fbank_results)
        np.save(os.path.join(args.data_path, "tgt_scores.npy"), tgt_results)

    # plot AUC
    if args.fig_path is not None:
        save_plot(thresholds, fbank_results, tgt_results, args.fig_path)


def cli_main():
    """
    Size metric measures Compactness of explanation. For a definition of this
    property, see:
    `"From Anecdotal Evidence to Quantitative Evaluation Methods:A Systematic
    Review on Evaluating Explainable AI" <https://dl.acm.org/doi/10.1145/3583558>`.

    Specifically, we measure the percentage of features which exceed a certain threshold.
    The threshold is defined iteratively from the minimum to the maximum value.
    The smaller the Size score, the  more compact is the explanation, hence the better.

    Variants of Size are used both in Explainability for image classification
    (see `"Explaining Image Classifiers Using Statistical Fault Localization"
    <https://link.springer.com/chapter/10.1007/978-3-030-58604-1_24>`), and
    speech recognition (see `"Explanations for Automatic Speech Recognition"
    <https://ieeexplore.ieee.org/document/10094635>`).
    """
    parser = argparse.ArgumentParser(description="Calculate Size metric of explanations.")
    parser.add_argument(
        "--explanation-path",
        required=True,
        type=str,
        help="Path to the h5 file where explanations are stored.")
    parser.add_argument(
        "--normalizer",
        default=[],
        nargs='+',
        choices=NORMALIZATION_REGISTRY.keys(),
        help="Normalizations to be applied to explanations.")
    parser.add_argument(
        "--step",
        type=int,
        default=5,
        help="Step size according to which computing size.")
    parser.add_argument(
        "--start-threshold",
        type=int,
        default=0,
        help="Value from which the thresholds are computed.")
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to the folder where npy arrays containing the data to plot will be saved.")
    parser.add_argument(
        "--fig-path",
        type=str,
        default=None,
        help="Path to the file where the plot is saved.")
    args = parser.parse_args()
    _main(args)


if __name__ == "__main__":
    cli_main()
