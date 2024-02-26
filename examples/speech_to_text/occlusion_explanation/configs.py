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

from os import path as op
import logging
from typing import Union
from dataclasses import dataclass, field
import yaml

from examples.speech_to_text.occlusion_explanation.perturbators import (
    get_perturbator,
    OcclusionFbankPerturbator,
    OcclusionDecoderEmbeddingsPerturbator)
from examples.speech_to_text.occlusion_explanation.scorers import get_scorer, Scorer
from fairseq.dataclass import FairseqDataclass

from fairseq.dataclass.utils import gen_parser_from_dataclass


LOGGER = logging.getLogger(__name__)


class PerturbConfig(object):
    """
    Wrapper class for perturbation config YAML
    """
    def __init__(self, yaml_path):
        self.perturb_cfg = {}
        if op.isfile(yaml_path):
            try:
                with open(yaml_path) as f:
                    self.perturb_cfg = yaml.load(f, Loader=yaml.FullLoader)
            except Exception as e:
                LOGGER.info(f"Failed to load config from {yaml_path}: {e}.")
        else:
            LOGGER.info(f"Cannot find {yaml_path}.")

    def get_perturbator_from_config(
            self,
            occlusion_choice
        ) -> Union[OcclusionFbankPerturbator, OcclusionDecoderEmbeddingsPerturbator]:
        if occlusion_choice == "fbank_occlusion":
            category = "continuous_fbank"
            if "fbank_occlusion" not in self.perturb_cfg:
                LOGGER.info("'fbank_occlusion' not in yaml file. Default values will be used.")
        elif occlusion_choice == "decoder_occlusion":
            category = "continuous_embed"
            if "decoder_occlusion" not in self.perturb_cfg:
                LOGGER.info("'decoder_occlusion' not in yaml file. Default values will be used.")
        else:
            raise ValueError("Invalid value for 'occlusion_choice'.")
        perturbation_category = self.perturb_cfg.get(occlusion_choice, {}).get("category", category)
        perturbation_class = get_perturbator(perturbation_category)
        return perturbation_class.from_config_dict(self.perturb_cfg)

    def get_scorer_from_config(self) -> Scorer:
        scorer_category = self.perturb_cfg.get("scorer", "predicted_token_diff")
        assert scorer_category == "predicted_token_diff" or scorer_category == "KL", \
            "Invalid value for 'occlusion_choice'. Must be one of ['fbank_occlusion', 'decoder_occlusion']."
        scorer_class = get_scorer(scorer_category)
        return scorer_class()


@dataclass
class OcclusionPerturbationConfig(FairseqDataclass):
    perturb_config: str = field(default=None, metadata={
        "help": "Filename of the yaml file containing the configuration for the occlusion perturbation."})
    n_masks: int = field(default=8000, metadata={"help": "Number of masks to be used for each instance."})
    original_probs: str = field(default=None, metadata={"help": "h5 file where original probabilities are stored."})
    save_file: str = field(default=None, metadata={"help": "File where heatmaps will be saved."})


def add_occlusion_perturbation_args(parser):
    group = parser.add_argument_group("occlusion_perturbation")
    # fmt: off
    gen_parser_from_dataclass(group, OcclusionPerturbationConfig())
    # fmt: on
    return group
