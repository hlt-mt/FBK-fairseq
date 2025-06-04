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
import os.path as op
import csv
from typing import Dict, List, Optional, Tuple

import torch

from examples.speech_to_text.data.speech_to_text_dataset_with_src import SpeechToTextDatasetWithSrc, \
    S2TDataConfigSrc, SpeechToTextDatasetCreatorWithSrc
from fairseq.data import Dictionary, ConcatDataset, ResamplingDataset, data_utils as fairseq_data_utils

from fairseq.data.audio.speech_to_text_dataset import _collate_frames

logger = logging.getLogger(__name__)


class SpeechToTextDatasetASRST(SpeechToTextDatasetWithSrc):
    """
        Extended version of SpeechToTextDatasetWithSrc that returns both target and source texts,
        in this case, translation and transcript.

        Main changes from parent class:
        - Adds check_src_lang_tag() method for source language tag validation
        - Returns 5-tuple instead of 4-tuple in __getitem__ (adds lang_prepended_transcript)
        - Enhanced collater() method that handles language-prepended transcripts
        - Uses language-prepended transcripts in prev_transcript_tokens for net_input
    """
    def __init__(
        self,
        split: str,
        is_train_split: bool,
        data_cfg: S2TDataConfigSrc,
        audio_paths: List[str],
        n_frames: List[int],
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
        bpe_tokenizer_src=None,
    ):
        super().__init__(
            split, is_train_split, data_cfg, audio_paths, n_frames, src_texts, tgt_texts, speakers,
            src_langs, tgt_langs, ids, tgt_dict, src_dict, pre_tokenizer, bpe_tokenizer,
            bpe_tokenizer_src)
        # NEW: Add source language tag validation (not present in parent class)
        self.check_src_lang_tag()

        logger.info(self.__repr__())

    def check_src_lang_tag(self):
        # Validate that source language tags are present in the source dictionary
        if self.data_cfg.prepend_tgt_lang_tag:
            assert self.src_langs is not None and self.src_dict is not None
            src_lang_tags = [
                self.LANG_TAG_TEMPLATE.format(t) for t in set(self.src_langs)
            ]
            assert all(t in self.src_dict for t in src_lang_tags)

    def __getitem__(self, index: int) -> Tuple[
        int, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Returns 5-tuple instead of 4-tuple (adds lang_prepended_transcript)
        index, source, target, transcript = super().__getitem__(index)

        lang_prepended_transcript = None
        if self.src_texts is not None:
            tokenized = self.tokenize_text_src(self.src_texts[index])
            lang_prepended_transcript = self.src_dict.encode_line(
                tokenized, add_if_not_exist=False, append_eos=True
            ).long()
            src_lang_tag = self.LANG_TAG_TEMPLATE.format(self.src_langs[index])
            src_lang_tag_idx = self.src_dict.index(src_lang_tag)
            lang_prepended_transcript = torch.cat(
                (torch.LongTensor([src_lang_tag_idx]), lang_prepended_transcript), 0)

        return index, source, target, transcript, lang_prepended_transcript

    def collater(self, samples: List[Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]]) -> Dict:
        # Updated collater to handle 5-tuple samples instead of 4-tuple
        if len(samples) == 0:
            return {}
        indices = torch.tensor([i for i, _, _, _, _ in samples], dtype=torch.long)
        frames = _collate_frames(
            [s for _, s, _, _, _ in samples], self.data_cfg.use_audio_input
        )
        # sort samples by descending number of frames
        n_frames = torch.tensor([s.size(0) for _, s, _, _, _ in samples], dtype=torch.long)
        n_frames, order = n_frames.sort(descending=True)
        indices = indices.index_select(0, order)
        frames = frames.index_select(0, order)

        target, target_lengths = None, None
        prev_output_tokens = None
        ntokens = None
        if self.tgt_texts is not None:
            target = fairseq_data_utils.collate_tokens(
                [t for _, _, t, _, _ in samples],
                self.tgt_dict.pad(),
                self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=False,
            )
            target = target.index_select(0, order)
            target_lengths = torch.tensor(
                [t.size(0) for _, _, t, _, _ in samples], dtype=torch.long
            ).index_select(0, order)
            prev_output_tokens = fairseq_data_utils.collate_tokens(
                [t for _, _, t, _, _ in samples],
                self.tgt_dict.pad(),
                self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, order)
            ntokens = sum(t.size(0) for _, _, t, _, _ in samples)

        # Source transcripts
        transcript, transcript_lengths = None, None
        prev_transcript_tokens = None
        ntokens_transcript = None
        if self.src_texts is not None:
            transcript = fairseq_data_utils.collate_tokens(
                [t for _, _, _, t, _ in samples],
                self.src_dict.pad(),
                self.src_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=False,
            )
            transcript = transcript.index_select(0, order)
            transcript_lengths = torch.tensor(
                [t.size(0) for _, _, _, t, _ in samples], dtype=torch.long
            ).index_select(0, order)
            prev_transcript_tokens = fairseq_data_utils.collate_tokens(
                [t for _, _, _, t, _ in samples],
                self.src_dict.pad(),
                self.src_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=True,
            )

        # Language-prepended source transcripts
        lang_prepended_transcript, lang_prepended_transcript_lengths = None, None
        prev_lang_prepended_transcript_tokens = None
        ntokens_lang_prepended_transcript = None
        if self.src_texts is not None:
            lang_prepended_transcript = fairseq_data_utils.collate_tokens(
                [t for _, _, _, _, t in samples],
                self.src_dict.pad(),
                self.src_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=False,
            )
            lang_prepended_transcript = lang_prepended_transcript.index_select(0, order)
            lang_prepended_transcript_lengths = torch.tensor(
                [t.size(0) for _, _, _, _, t in samples], dtype=torch.long
            ).index_select(0, order)
            prev_lang_prepended_transcript_tokens = fairseq_data_utils.collate_tokens(
                [t for _, _, _, _, t in samples],
                self.src_dict.pad(),
                self.src_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=True,
            )
            prev_lang_prepended_transcript_tokens = (
                prev_lang_prepended_transcript_tokens.index_select(0, order))
            ntokens_lang_prepended_transcript = sum(t.size(0) for _, _, _, _, t in samples)

        out = {
            "id": indices,
            "net_input": {
                "src_tokens": frames,
                "src_lengths": n_frames,
                "prev_output_tokens": prev_output_tokens,
                "prev_transcript_tokens": prev_lang_prepended_transcript_tokens,
            },
            "target": target,
            "target_lengths": target_lengths,
            "transcript": transcript,
            "transcript_lengths": transcript_lengths,
            "prepended_transcript": lang_prepended_transcript,
            "prepended_transcript_lengths": lang_prepended_transcript_lengths,
            "ntokens": ntokens,
            "ntokens_transcript": ntokens_transcript,
            "ntokens_prepended_transcript": ntokens_lang_prepended_transcript,
            "nsentences": len(samples),
        }
        return out


class SpeechToTextDatasetCreatorASRST(SpeechToTextDatasetCreatorWithSrc):
    """
    Same as SpeechToTextDatasetCreatorWithSrc but returning SpeechToTextDatasetASRST instead.
    """
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
        bpe_tokenizer_src,
    ) -> SpeechToTextDatasetASRST:
        audio_paths, n_frames, src_texts, tgt_texts, ids = [], [], [], [], []
        speakers, src_langs, tgt_langs = [], [], []
        for s in samples:
            ids.extend([ss[cls.KEY_ID] for ss in s])
            audio_paths.extend(
                [op.join(data_cfg.audio_root, ss[cls.KEY_AUDIO]) for ss in s]
            )
            n_frames.extend([int(ss[cls.KEY_N_FRAMES]) for ss in s])
            tgt_texts.extend([ss[cls.KEY_TGT_TEXT] for ss in s])
            src_texts.extend([ss.get(cls.KEY_SRC_TEXT, cls.DEFAULT_SRC_TEXT) for ss in s])
            speakers.extend([ss.get(cls.KEY_SPEAKER, cls.DEFAULT_SPEAKER) for ss in s])
            src_langs.extend([ss.get(cls.KEY_SRC_LANG, cls.DEFAULT_LANG) for ss in s])
            tgt_langs.extend([ss.get(cls.KEY_TGT_LANG, cls.DEFAULT_LANG) for ss in s])
        return SpeechToTextDatasetASRST(
            split_name,
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
            src_dict,
            pre_tokenizer,
            bpe_tokenizer,
            bpe_tokenizer_src
        )
