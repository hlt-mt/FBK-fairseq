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

import logging
from typing import Dict

from examples.speech_to_text.occlusion_explanation.aggregators import get_aggregator, AGGREGATION_REGISTRY
from examples.speech_to_text.occlusion_explanation.utils import read_feature_attribution_maps_from_h5
from examples.speech_to_text.tasks.speech_to_text_ctc import SpeechToTextCtcTask
from examples.speech_to_text.xai_metrics.deletion_insertion_dataset import \
    FeatureAttributionEvaluationSpeechToTextDataset
from fairseq.tasks import register_task


LOGGER = logging.getLogger(__name__)


@register_task("feature_attribution_evaluation_task")
class FeatureAttributionEvaluationTask(SpeechToTextCtcTask):
    @staticmethod
    def add_args(parser):
        SpeechToTextCtcTask.add_args(parser)
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
            "--explanation-path",
            default=None,
            help="Path of the h5 file where heatmaps are saved.")
        parser.add_argument(
            "--aggregator",
            default="sentence_aggregator_no_norm",
            choices=AGGREGATION_REGISTRY.keys(),
            help="Aggregator type to obtain sentence-level explanations.")

    def __init__(self, args, tgt_dict, src_dict):
        super().__init__(args, tgt_dict, src_dict)
        # initialize sentence-level aggregator
        aggregator_class = get_aggregator(args.aggregator)
        aggregator = aggregator_class()
        # get aggregated heatmaps
        explanations = read_feature_attribution_maps_from_h5(self.args.explanation_path)
        self.aggregated_explanations = aggregator(explanations)
        # Size of the percentage intervals according to which insertion/deletion of input elements is performed
        self.interval_size = int(self.args.perc_interval)
        assert 100 % self.interval_size == 0, "100 must be a multiple of --perc-interval."
        # number of num_intervals in which insertion/deletion of input elements is performed
        self.num_intervals = (100 // self.interval_size) + 1

    # creating the perturbed dataset for the metric
    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        super().load_dataset(split, epoch=1, combine=False, **kwargs)
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
