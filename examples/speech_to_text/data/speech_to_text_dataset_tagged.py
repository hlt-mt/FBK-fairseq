# Copyright 2021 FBK

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
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F

from examples.speech_to_text.data.speech_to_text_dataset_with_src import S2TDataConfigSrc, \
    SpeechToTextDatasetCreatorWithSrc, SpeechToTextDatasetWithSrc
from fairseq.data import (
    Dictionary,
    data_utils as fairseq_data_utils,
)
from fairseq.data.audio.speech_to_text_dataset import _collate_frames


logger = logging.getLogger(__name__)


class S2TDataConfigTagged(S2TDataConfigSrc):
    """Wrapper class for data config YAML"""
    def __init__(self, yaml_path):
        super().__init__(yaml_path)

    @property
    def tags(self):
        """list of tag names"""
        return self.config.get("tags", [])


class SpeechToTextDatasetTagged(SpeechToTextDatasetWithSrc):

    def __init__(
        self,
        split: str,
        is_train_split: bool,
        data_cfg: S2TDataConfigTagged,
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
        super().__init__(split, is_train_split, data_cfg, audio_paths, n_frames,
                         src_texts, tgt_texts, speakers, src_langs, tgt_langs,
                         ids, tgt_dict, src_dict, pre_tokenizer, bpe_tokenizer, bpe_tokenizer_src)
        self.tgt_initial_tags_idxs = []
        self.tgt_end_tags_idxs = []
        self.src_initial_tags_idxs = []
        self.src_end_tags_idxs = []
        for tag in self.data_cfg.tags:
            init_tag_idx = tgt_dict.index("<{}>".format(tag))
            end_tag_idx = tgt_dict.index("</{}>".format(tag))
            assert init_tag_idx != tgt_dict.unk_index, "<{}> was not found in the tgt dict".format(tag)
            assert end_tag_idx != tgt_dict.unk_index, "</{}> was not found in the tgt dict".format(tag)
            self.tgt_initial_tags_idxs.append(init_tag_idx)
            self.tgt_end_tags_idxs.append(end_tag_idx)

            init_tag_idx = src_dict.index("<{}>".format(tag))
            end_tag_idx = src_dict.index("</{}>".format(tag))
            assert init_tag_idx != src_dict.unk_index, "<{}> was not found in the src dict".format(tag)
            assert end_tag_idx != src_dict.unk_index, "</{}> was not found in the src dict".format(tag)
            self.src_initial_tags_idxs.append(init_tag_idx)
            self.src_end_tags_idxs.append(end_tag_idx)

    @staticmethod
    def strip_tags_from_text(tokens, initial_tag_idxs, end_tag_idxs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Removes from the text the tags and returns:
          - a Tensor with the tokens representing the clean text
          - a Tensor indicating the ID of the tag each token belongs to (or 0 if it belongs to no tag)
        """
        clean_tokens = []
        tags = []
        current_tag_idx = None
        for idx_t in tokens:
            idx = idx_t.item()
            if idx in initial_tag_idxs:
                current_tag_idx = initial_tag_idxs.index(idx)
            elif current_tag_idx is not None and end_tag_idxs[current_tag_idx] == idx:
                current_tag_idx = None
            else:
                clean_tokens.append(idx)
                tags.append(current_tag_idx + 1 if current_tag_idx is not None else 0)
        return torch.tensor(clean_tokens).to(tokens), torch.tensor(tags).to(tokens)

    def __getitem__(
        self, index: int
    ) -> Tuple[int, Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        index, source, target, transcript = super().__getitem__(index)

        clean_target, tags_target = self.strip_tags_from_text(
            target, self.tgt_initial_tags_idxs, self.tgt_end_tags_idxs)
        clean_transcript, tags_transcript = self.strip_tags_from_text(
            transcript, self.src_initial_tags_idxs, self.src_end_tags_idxs)

        return index, source, clean_target, clean_transcript, tags_target, tags_transcript

    def collater(
        self,
        samples: List[Tuple[int, Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]]]
    ) -> Dict:
        if len(samples) == 0:
            return {}
        indices = torch.tensor([i for i, _, _, _, _, _ in samples], dtype=torch.long)
        frames = _collate_frames(
            [s for _, s, _, _, _, _ in samples], self.data_cfg.use_audio_input
        )
        # sort samples by descending number of frames
        n_frames = torch.tensor([s.size(0) for _, s, _, _, _, _ in samples], dtype=torch.long)
        n_frames, order = n_frames.sort(descending=True)
        indices = indices.index_select(0, order)
        frames = frames.index_select(0, order)

        target, target_lengths = None, None
        prev_output_tokens = None
        ntokens = None
        if self.tgt_texts is not None:
            target = fairseq_data_utils.collate_tokens(
                [t for _, _, t, _, _, _ in samples],
                self.tgt_dict.pad(),
                self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=False,
            )
            target = target.index_select(0, order)
            target_lengths = torch.tensor(
                [t.size(0) for _, _, t, _, _, _ in samples], dtype=torch.long
            ).index_select(0, order)
            prev_output_tokens = fairseq_data_utils.collate_tokens(
                [t for _, _, t, _, _, _ in samples],
                self.tgt_dict.pad(),
                self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, order)
            ntokens = sum(t.size(0) for _, _, t, _, _, _ in samples)

        # Source transcripts
        transcript, transcript_lengths = None, None
        prev_transcript_tokens = None
        ntokens_transcript = None
        if self.src_texts is not None:
            transcript = fairseq_data_utils.collate_tokens(
                [t for _, _, _, t, _, _ in samples],
                self.src_dict.pad(),
                self.src_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=False,
            )
            transcript = transcript.index_select(0, order)
            transcript_lengths = torch.tensor(
                [t.size(0) for _, _, _, t, _, _ in samples], dtype=torch.long
            ).index_select(0, order)
            prev_transcript_tokens = fairseq_data_utils.collate_tokens(
                [t for _, _, _, t, _, _ in samples],
                self.src_dict.pad(),
                self.src_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=True,
            )
            prev_transcript_tokens = prev_transcript_tokens.index_select(0, order)
            ntokens_transcript = sum(t.size(0) for _, _, _, t, _, _ in samples)

        pad_idx = 0
        tgt_len = target.shape[1]
        tgt_tags = []
        for _, _, _, _, tgt_tag, _ in samples:
            tgt_tags.append(F.pad(tgt_tag, (0, tgt_len - tgt_tag.shape[0]), value=pad_idx))
        tgt_tags = torch.stack(tgt_tags).index_select(0, order)
        src_len = transcript.shape[1]
        src_tags = []
        for _, _, _, _, _, src_tag in samples:
            src_tags.append(F.pad(src_tag, (0, src_len - src_tag.shape[0]), value=pad_idx))
        src_tags = torch.stack(src_tags).index_select(0, order)

        out = {
            "id": indices,
            "net_input": {
                "src_tokens": frames,
                "src_lengths": n_frames,
                "prev_output_tokens": prev_output_tokens,
                "prev_transcript_tokens": prev_transcript_tokens,
                "prev_target_tags": torch.roll(tgt_tags, 1),
                "prev_transcript_tags": torch.roll(src_tags, 1),
            },
            "target": target,
            "target_lengths": target_lengths,
            "target_tags": tgt_tags,
            "transcript": transcript,
            "transcript_lengths": transcript_lengths,
            "transcript_tags": src_tags,
            "ntokens": ntokens,
            "ntokens_transcript": ntokens_transcript,
            "nsentences": len(samples),
        }
        return out


class SpeechToTextDatasetTaggedCreator(SpeechToTextDatasetCreatorWithSrc):
    @classmethod
    def _from_list(
        cls,
        split_name: str,
        is_train_split,
        samples: List[List[Dict]],
        data_cfg: S2TDataConfigTagged,
        tgt_dict,
        src_dict,
        pre_tokenizer,
        bpe_tokenizer,
        bpe_tokenizer_src,
    ) -> SpeechToTextDatasetTagged:
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
        return SpeechToTextDatasetTagged(
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
