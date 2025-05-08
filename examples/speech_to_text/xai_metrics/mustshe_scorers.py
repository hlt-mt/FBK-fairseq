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

from abc import ABC, abstractmethod
import logging
from typing import Dict, List
from examples.speech_to_text.scripts.gender.mustshe_gender_accuracy \
    import sentence_level_statistics, MuSTSheEntry, GenderTerms


LOGGER = logging.getLogger(__name__)

MUSTSHE_SCORER_REGISTRY = {}
MUSTSHE_SCORER_CLASS_NAMES = set()


class MuSTSheScorer(ABC):
    def __init__(self):
        self.pred = []
        self.mustshe = []

        try:
            from sacremoses import MosesTokenizer
            self.moses_tokenizer = MosesTokenizer()
        except ImportError:
            raise ImportError("Please install sacremoses by running 'pip install sacremoses'.")

    def _extract_ref_mustshe_hyp(self, line: Dict) -> MuSTSheEntry:
        """
        Extracts relevant information from the hypotheses the model originally generated 
        from the MuST-SHE dataset and that we want to explain.
        """   
        category = line['category']
        gender_terms = [GenderTerms(*line['found_term_pairs'].strip().lower().split(" "))]
        return MuSTSheEntry(category, gender_terms)

    def add_string(self, ref: Dict, pred: str):
        mustshe_entry = self._extract_ref_mustshe_hyp(ref)
        self.mustshe.append(mustshe_entry)
        self.pred.append(self.moses_tokenizer.tokenize(pred.strip().lower(), return_str=True))

    @abstractmethod
    def score(self, categories: List[str] = ["Global"]) -> float:
        pass


def register_scorer(name):
    def register_scorer_cls(cls):
        if name in MUSTSHE_SCORER_REGISTRY:
            raise ValueError(
                f"Cannot register duplicate scorer ({name})")
        if not issubclass(cls, MuSTSheScorer):
            raise ValueError(
                f"Scorer ({name}: {cls.__name__}) must extend MuSTSheScorer")
        if cls.__name__ in MUSTSHE_SCORER_CLASS_NAMES:
            raise ValueError(
                f"Cannot register scorer with duplicate class name ({cls.__name__})")
        MUSTSHE_SCORER_REGISTRY[name] = cls
        MUSTSHE_SCORER_CLASS_NAMES.add(cls.__name__)
        LOGGER.debug(f"Scorer registered: {name}.")
        return cls
    return register_scorer_cls


def build_scorer(name):
    return MUSTSHE_SCORER_REGISTRY[name]()


@register_scorer("gender_accuracy")
class GenderAccuracyScorer(MuSTSheScorer):
    """
    Computes the percentage of times the generated gender term corresponds to the reference 
    gender term as annotated in the MuST-SHE dataset. This is computed only for in-coverage terms.
    """

    def score(self, categories: List[str] = ["Global"]) -> float:
        self.sentence_scores = sentence_level_statistics(self.pred, self.mustshe)
        assert len(self.sentence_scores) == len(self.mustshe)
        tot_correct = 0
        tot_wrong = 0
        for sentence_score, mustshe_entry in zip(self.sentence_scores, self.mustshe):
            if "Global" not in categories and mustshe_entry.category not in categories:
                continue
            tot_correct += sentence_score["num_correct"]
            tot_wrong += sentence_score["num_wrong"]
        if tot_correct + tot_wrong > 0:
            return float(tot_correct) / float(tot_correct + tot_wrong)
        else:
            return 0.0


@register_scorer("gender_flip_rate")
class GenderFlipRateScorer(GenderAccuracyScorer):
    """
    In the scenario of input perturbation, computes the percentage of times the model generates the opposite 
    gender compared to the one it originally had generated in the unperturbed scenario.
    """

    def _extract_ref_mustshe_hyp(self, line: Dict) -> MuSTSheEntry:
        """
        Extracts relevant information from the hypotheses the model originally generated
        from the MuST-SHE dataset and that we want to explain.
        """
        category = line['category']
        correct, wrong = line['found_term_pairs'].strip().lower().split(" ")
        found = line['found_terms'].lower()
        not_found = correct if found == wrong else wrong
        # The flip rate can be computed just like the gender accuracy but instead of 
        # correct / (correct + wrong), the score is not_found / (not_found + found).
        gender_terms = [GenderTerms(not_found, found)]
        return MuSTSheEntry(category, gender_terms)


@register_scorer("gender_coverage")
class GenderCoverageScorer(MuSTSheScorer):
    """
    Computes the percentage of annotated gender terms that are present in the hypotheses.
    """

    def score(self, categories: List[str] = ["Global"]) -> float:
        self.sentence_scores = sentence_level_statistics(self.pred, self.mustshe)
        assert len(self.sentence_scores) == len(self.mustshe)
        tot_terms = 0
        tot_found = 0
        for sentence_score, mustshe_entry in zip(self.sentence_scores, self.mustshe):
            if "Global" not in categories and mustshe_entry.category not in categories:
                continue
            tot_terms += sentence_score["num_terms"]
            tot_found += sentence_score["num_terms_found"]
        if tot_terms > 0:
            return float(tot_found) / float(tot_terms)
        else:
            return 0.0
