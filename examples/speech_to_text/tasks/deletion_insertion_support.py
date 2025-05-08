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

import logging
from typing import Dict

from examples.speech_to_text.occlusion_explanation.aggregators import get_aggregator, AGGREGATION_REGISTRY
from examples.speech_to_text.occlusion_explanation.normalizers import NORMALIZATION_REGISTRY, get_normalizer
from examples.speech_to_text.occlusion_explanation.utils import read_feature_attribution_maps_from_h5
from examples.speech_to_text.xai_metrics.deletion_insertion_dataset import \
    FeatureAttributionEvaluationSpeechToTextDataset


LOGGER = logging.getLogger(__name__)


class FeatureAttributionEvaluationSupport:
    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--metric",
            default="deletion",
            choices=['insertion', 'deletion'],
            help="Type of metric to be used, to be chosen among ['insertion', 'deletion'].")
        parser.add_argument(
            "--perc-interval",
            default=5,
            help="Percentage interval indicating how often input features are removed/inserted.")
        parser.add_argument(
            "--max-percent",
            default=100)
        parser.add_argument(
            "--explanation-path",
            default=None,
            help="Path of the h5 file where heatmaps are saved.")
        parser.add_argument(
            "--aggregator",
            default="sentence",
            choices=AGGREGATION_REGISTRY.keys(),
            help="Aggregator type to obtain sentence-level explanations.")
        parser.add_argument(
            "--normalizer",
            default=[],
            nargs='+',
            choices=NORMALIZATION_REGISTRY.keys(),
            help="Normalizations to be applied to explanations.")

    def setup_evaluation(self, args):
        # initialize sentence-level aggregator
        aggregator_class = get_aggregator(args.aggregator)
        normalizers = [get_normalizer(norm) for norm in args.normalizer]
        aggregator = aggregator_class()
        # get aggregated heatmaps
        explanations = read_feature_attribution_maps_from_h5(self.args.explanation_path)
        for sample in explanations:
            for norm in normalizers:
                norm = norm()
                explanations[sample]["fbank_heatmap"], explanations[sample]["tgt_embed_heatmap"] = norm(
                    explanations[sample]["fbank_heatmap"], explanations[sample]["tgt_embed_heatmap"])
        self.aggregated_explanations = aggregator(explanations)
        # Size of the percentage intervals according to which insertion/deletion of input elements is performed
        self.interval_size = int(self.args.perc_interval)
        self.max_percent = int(self.args.max_percent)
        assert self.max_percent % self.interval_size == 0, "--max-percent must be a multiple of --perc-interval."
        # number of num_intervals in which insertion/deletion of input elements is performed
        self.num_intervals = (self.max_percent // self.interval_size) + 1

    # creating the perturbed dataset for the metric
    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        super().load_dataset(split, epoch=epoch, combine=combine, **kwargs)
        metric_dataset = FeatureAttributionEvaluationSpeechToTextDataset(
            original_dataset=self.datasets[split],
            aggregated_fbank_heatmaps=self.aggregated_explanations,
            interval_size=self.interval_size,
            num_intervals=self.num_intervals,
            metric=self.args.metric)
        self.datasets[split] = metric_dataset

    def customize_sample_id(self, index: int, sample_id: int, sample: Dict) -> str:
        interval_id = sample_id % self.num_intervals
        return str(sample["orig_id"][index].item()) + "-" + str(interval_id)
