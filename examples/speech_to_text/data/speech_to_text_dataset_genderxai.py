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
import os.path as op
from typing import Dict, List, Optional

import torch

from examples.speech_to_text.data.speech_to_text_dataset_with_src import S2TDataConfigSrc, \
    SpeechToTextDatasetCreatorWithSrc, SpeechToTextDatasetWithSrc
from fairseq.data import Dictionary


logger = logging.getLogger(__name__)


class SpeechToTextDatasetGenderXai(SpeechToTextDatasetWithSrc):

    def __init__(
            self,
            split: str,
            is_train_split: bool,
            data_cfg: S2TDataConfigSrc,
            audio_paths: List[str],
            n_frames: List[int],
            found_terms: List[str],
            found_term_pairs: List[str],
            gender_terms_indices: List[int],
            swapped_tgt_texts: List[str],
            src_texts: Optional[List[str]] = None,
            tgt_texts: Optional[List[str]] = None,
            speakers: Optional[List[str]] = None,
            src_langs: Optional[List[str]] = None,
            tgt_langs: Optional[List[str]] = None,
            ids: Optional[List[str]] = None,
            tgt_dict: Optional[Dictionary] = None,
            src_dict: Optional[Dictionary] = None,
            pre_tokenizer=None,
            bpe_tokenizer=None,
            bpe_tokenizer_src=None):
        super().__init__(
            split, is_train_split, data_cfg, audio_paths, n_frames, src_texts, 
            tgt_texts, speakers, src_langs, tgt_langs, ids, tgt_dict, src_dict, 
            pre_tokenizer, bpe_tokenizer, bpe_tokenizer_src)

        assert len(found_terms) == self.n_samples
        assert len(found_term_pairs) == self.n_samples
        assert len(gender_terms_indices) == self.n_samples
        assert len(swapped_tgt_texts) == self.n_samples

        # Extra fields wrt SpeechToTextDatasetWithSrc
        self.found_terms = found_terms
        self.found_term_pairs = found_term_pairs
        self.gender_terms_indices = gender_terms_indices
        self.swapped_tgt_texts = swapped_tgt_texts

        logger.info(self.__repr__())

    def __getitem__(self, index: int) -> Dict:
        item = super().__getitem__(index)

        swapped_target = None
        if self.swapped_tgt_texts is not None:
            tokenized = self.tokenize_text(self.swapped_tgt_texts[index])
            swapped_target = self.tgt_dict.encode_line(
                tokenized, add_if_not_exist=False, append_eos=True
            ).long()
            if self.data_cfg.prepend_tgt_lang_tag:
                lang_tag = self.LANG_TAG_TEMPLATE.format(self.tgt_langs[index])
                lang_tag_idx = self.tgt_dict.index(lang_tag)
                target = torch.cat((torch.LongTensor([lang_tag_idx]), target), 0)
        
        return (
            *item,
            self.found_terms[index],
            self.found_term_pairs[index],
            self.gender_terms_indices[index],
            swapped_target)


class SpeechToTextDatasetCreatorGenderXai(SpeechToTextDatasetCreatorWithSrc):
    # Extra fields wrt SpeechToTextDatasetCreator.
    KEY_FOUND_TERMS, KEY_FOUND_TERM_PAIRS = "found_terms", "found_term_pairs"
    KEY_GENDER_TERMS_INDICES, KEY_SWAPPED_TGT_TEXT = "gender_terms_indices", "swapped_tgt_text"

    @classmethod
    def _from_list(
            cls,
            split_name: str,
            is_train_split,
            samples: List[List[Dict]],
            data_cfg: S2TDataConfigSrc,
            tgt_dict,
            src_dict,
            pre_tokenizer,
            bpe_tokenizer,
            bpe_tokenizer_src) -> SpeechToTextDatasetGenderXai:
        audio_paths, n_frames, src_texts, tgt_texts, ids = [], [], [], [], []
        speakers, src_langs, tgt_langs = [], [], []
        found_terms, found_term_pairs, gender_terms_indices, swapped_tgt_texts = [], [], [], []
        for s in samples:
            ids.extend([ss[cls.KEY_ID] for ss in s])
            audio_paths.extend(
                [op.join(data_cfg.audio_root, ss[cls.KEY_AUDIO]) for ss in s])
            n_frames.extend([int(ss[cls.KEY_N_FRAMES]) for ss in s])
            tgt_texts.extend([ss[cls.KEY_TGT_TEXT] for ss in s])
            src_texts.extend([ss.get(cls.KEY_SRC_TEXT, cls.DEFAULT_SRC_TEXT) for ss in s])
            speakers.extend([ss.get(cls.KEY_SPEAKER, cls.DEFAULT_SPEAKER) for ss in s])
            src_langs.extend([ss.get(cls.KEY_SRC_LANG, cls.DEFAULT_LANG) for ss in s])
            tgt_langs.extend([ss.get(cls.KEY_TGT_LANG, cls.DEFAULT_LANG) for ss in s])
            found_terms.extend([ss[cls.KEY_FOUND_TERMS] for ss in s])
            found_term_pairs.extend([ss[cls.KEY_FOUND_TERM_PAIRS] for ss in s])
            gender_terms_indices.extend([ss[cls.KEY_GENDER_TERMS_INDICES] for ss in s])
            swapped_tgt_texts.extend([ss[cls.KEY_SWAPPED_TGT_TEXT] for ss in s])
        return SpeechToTextDatasetGenderXai(
            split_name,
            is_train_split,
            data_cfg,
            audio_paths,
            n_frames,
            found_terms,
            found_term_pairs,
            gender_terms_indices,
            swapped_tgt_texts,
            src_texts,
            tgt_texts,
            speakers,
            src_langs,
            tgt_langs,
            ids,
            tgt_dict,
            src_dict,
            pre_tokenizer,
            bpe_tokenizer,
            bpe_tokenizer_src)
