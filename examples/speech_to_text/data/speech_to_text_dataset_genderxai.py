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
from typing import Dict, List, Optional, Tuple, Union

import torch

from fairseq.data import (
    Dictionary,
    data_utils as fairseq_data_utils)
from fairseq.data.audio.speech_to_text_dataset import S2TDataConfig, SpeechToTextDataset, \
    SpeechToTextDatasetCreator, _collate_frames


logger = logging.getLogger(__name__)


class SpeechToTextDatasetGenderXai(SpeechToTextDataset):

    def __init__(
        self,
        split: str,
        is_train_split: bool,
        data_cfg: S2TDataConfig,
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
        pre_tokenizer=None,
        bpe_tokenizer=None
    ):
        super().__init__(
            split,
            is_train_split,
            data_cfg,
            audio_paths,
            n_frames,
            src_texts,
            tgt_texts,
            speakers,
            src_langs,
            tgt_langs,
            ids,
            tgt_dict,
            pre_tokenizer,
            bpe_tokenizer)

        assert len(found_terms) == self.n_samples
        assert len(found_term_pairs) == self.n_samples
        assert len(gender_terms_indices) == self.n_samples
        assert len(swapped_tgt_texts) == self.n_samples

        # Extra fields wrt SpeechToTextDataset
        self.found_terms = found_terms
        self.found_term_pairs = found_term_pairs
        self.gender_terms_indices = gender_terms_indices
        self.swapped_tgt_texts = swapped_tgt_texts

        logger.info(self.__repr__())

    def __getitem__(self, index: int) -> Tuple[
            int,
            torch.Tensor,
            Union[torch.Tensor, None],
            List[str],
            List[str],
            List[str],
            torch.Tensor]:
        item = super().__getitem__(index)   # contains (index, source, target)

        # If there is no target
        if len(item) == 2:
            item = item + (None,)

        # Get the token indices for the swapped target text
        tokenized = self.tokenize_text(self.swapped_tgt_texts[index])
        swapped_tgt_tokens = self.tgt_dict.encode_line(
            tokenized, add_if_not_exist=False, append_eos=True
        ).long()
        gender_term_indices = self.gender_terms_indices[index]
        if self.data_cfg.prepend_tgt_lang_tag:
            lang_tag = self.LANG_TAG_TEMPLATE.format(self.tgt_langs[index])
            lang_tag_idx = self.tgt_dict.index(lang_tag)
            swapped_tgt_tokens = torch.cat((torch.LongTensor([lang_tag_idx]), swapped_tgt_tokens), 0)
            # Since we prepend a token, we need to adjust the indices of the gender term by 1
            start, end = gender_term_indices.split('-')
            start, end = int(start), int(end)
            gender_term_indices = f"{start + 1}-{end + 1}"
        
        return (
            *item,
            self.found_terms[index],
            self.found_term_pairs[index],
            gender_term_indices,
            swapped_tgt_tokens)
    
    def collater(self, samples: List[Tuple[
         int, torch.Tensor, Union[torch.Tensor, None], List[str], List[str], List[str], torch.Tensor]]) -> Dict:
        
        if len(samples) == 0:
            return {}
        
        indices = torch.tensor([i for i, _, _, _, _, _, _ in samples], dtype=torch.long)
        frames = _collate_frames(
            [s for _, s, _, _, _, _, _ in samples], self.data_cfg.use_audio_input)
        # sort samples by descending number of frames
        n_frames = torch.tensor([s.size(0) for _, s, _, _, _, _, _ in samples], dtype=torch.long)
        n_frames, order = n_frames.sort(descending=True)
        indices = indices.index_select(0, order)
        frames = frames.index_select(0, order)

        target, target_lengths = None, None
        prev_output_tokens = None
        ntokens = None
        if self.tgt_texts is not None:
            target = fairseq_data_utils.collate_tokens(
                [t for _, _, t, _, _, _, _ in samples],
                self.tgt_dict.pad(),
                self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=False)
            target = target.index_select(0, order)
            target_lengths = torch.tensor(
                [t.size(0) for _, _, t, _, _, _, _ in samples], dtype=torch.long
            ).index_select(0, order)
            prev_output_tokens = fairseq_data_utils.collate_tokens(
                [t for _, _, t, _, _, _, _ in samples],
                self.tgt_dict.pad(),
                self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=True)
            prev_output_tokens = prev_output_tokens.index_select(0, order)
            ntokens = sum(t.size(0) for _, _, t, _, _, _, _ in samples)

        # Extra fields for gender analysis
        found_terms = [ft for _, _, _, ft, _, _, _ in samples]
        found_terms = [found_terms[i] for i in order]
        found_term_pairs = [ftp for _, _, _, _, ftp, _, _ in samples]
        found_term_pairs = [found_term_pairs[i] for i in order]
        
        swapped_target = fairseq_data_utils.collate_tokens(
            [stt for _, _, _, _, _, _, stt in samples],
            self.tgt_dict.pad(),
            self.tgt_dict.eos(),
            left_pad=False,
            move_eos_to_beginning=False)
        swapped_target = swapped_target.index_select(0, order)
        swapped_tgt_lengths = torch.tensor(
                [stt.size(0) for _, _, _, _, _, _, stt in samples], dtype=torch.long).index_select(0, order)
        prev_swapped_tokens = fairseq_data_utils.collate_tokens(
            [stt for _, _, _, _, _, _, stt in samples],
            self.tgt_dict.pad(),
            self.tgt_dict.eos(),
            left_pad=False,
            move_eos_to_beginning=True)
        prev_swapped_tokens = prev_swapped_tokens.index_select(0, order)

        gender_term_starts = [int(gti.split('-')[0]) for _, _, _, _, _, gti, _ in samples]
        gender_term_ends = [int(gti.split('-')[1]) for _, _, _, _, _, gti, _ in samples]
        swapped_term_ends = gender_term_ends.copy()
        # We retrieve the index of the last token in the swapped term from the difference
        # between the swapped target length and the original target length
        for i, (_, _, tgts, _, _, _, swapped_tgts) in enumerate(samples):
            swapped_term_ends[i] += swapped_tgts.size(0) - tgts.size(0)
        gender_term_starts = torch.LongTensor([gender_term_starts[i] for i in order])
        gender_term_ends = torch.LongTensor([gender_term_ends[i] for i in order])
        swapped_term_ends = torch.LongTensor([swapped_term_ends[i] for i in order])

        out = {
            "id": indices,
            "net_input": {
                "src_tokens": frames,
                "src_lengths": n_frames,
                "prev_output_tokens": prev_output_tokens,
                "swapped_prev_output_tokens": prev_swapped_tokens
            },
            "target": target,
            "target_lengths": target_lengths,
            "swapped_target": swapped_target,
            "swapped_target_lengths": swapped_tgt_lengths,
            "ntokens": ntokens,
            "nsentences": len(samples),
            "found_terms": found_terms,
            "found_term_pairs": found_term_pairs,
            "gender_term_starts": gender_term_starts,
            "gender_term_ends": gender_term_ends,
            "swapped_term_ends": swapped_term_ends}
        return out


class SpeechToTextDatasetCreatorGenderXai(SpeechToTextDatasetCreator):
    # Extra fields wrt SpeechToTextDatasetCreator.
    KEY_FOUND_TERMS, KEY_FOUND_TERM_PAIRS = "found_terms", "found_term_pairs"
    KEY_GENDER_TERMS_INDICES, KEY_SWAPPED_TGT_TEXT = "gender_terms_indices", "swapped_tgt_text"

    @classmethod
    def _from_list(
        cls,
        split_name: str,
        is_train_split,
        samples: List[List[Dict]],
        data_cfg: S2TDataConfig,
        tgt_dict,
        pre_tokenizer,
        bpe_tokenizer
    ) -> SpeechToTextDatasetGenderXai:
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
            pre_tokenizer,
            bpe_tokenizer)
