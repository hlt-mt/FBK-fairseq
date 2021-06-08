# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import csv
import logging
import os.path as op
from typing import Dict, List, Optional, Tuple

import torch
from torch.nn import functional as F

from examples.speech_to_text.data.speech_to_text_dataset_with_src import S2TDataConfigSrc, SpeechToTextDatasetWithSrc
from fairseq.data import (
    ConcatDataset,
    Dictionary,
    ResamplingDataset,
    data_utils as fairseq_data_utils,
)
from fairseq.data.audio.feature_transforms import CompositeAudioFeatureTransform
from fairseq.data.audio.speech_to_text_dataset import SpeechToTextDatasetCreator, \
    _collate_frames, get_features_or_waveform

logger = logging.getLogger(__name__)


class SpeechToTextDatasetKD(SpeechToTextDatasetWithSrc):
    LANG_TAG_TEMPLATE = "<lang:{}>"

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
        idxs: Optional[List[str]] = None,
        probs: Optional[List[str]] = None,
        tgt_dict: Optional[Dictionary] = None,
        src_dict: Optional[Dictionary] = None,
        pre_tokenizer=None,
        bpe_tokenizer=None,
        bpe_tokenizer_src=None,
    ):
        super().__init__(split, is_train_split, data_cfg, audio_paths, n_frames, src_texts, tgt_texts, speakers,
                         src_langs, tgt_langs, ids, tgt_dict, src_dict, pre_tokenizer, bpe_tokenizer, bpe_tokenizer_src)
        self.split, self.is_train_split = split, is_train_split
        self.data_cfg = data_cfg
        self.audio_paths, self.n_frames = audio_paths, n_frames
        self.n_samples = len(audio_paths)
        assert len(n_frames) == self.n_samples > 0
        assert src_texts is None or len(src_texts) == self.n_samples
        assert tgt_texts is None or len(tgt_texts) == self.n_samples
        assert speakers is None or len(speakers) == self.n_samples
        assert src_langs is None or len(src_langs) == self.n_samples
        assert tgt_langs is None or len(tgt_langs) == self.n_samples
        assert ids is None or len(ids) == self.n_samples
        assert (tgt_dict is None and tgt_texts is None) or (
            tgt_dict is not None and tgt_texts is not None
        )
        assert (src_dict is None and src_texts is None) or (
                src_dict is not None and src_texts is not None
        )
        assert idxs is not None and probs is not None
        self.src_texts, self.tgt_texts = src_texts, tgt_texts
        self.src_langs, self.tgt_langs = src_langs, tgt_langs
        self.tgt_dict = tgt_dict
        self.src_dict = src_dict
        self.check_tgt_lang_tag()
        self.ids = ids
        self.shuffle = data_cfg.shuffle if is_train_split else False

        self.feature_transforms = CompositeAudioFeatureTransform.from_config_dict(
            self.data_cfg.get_feature_transforms(split, is_train_split)
        )

        self.pre_tokenizer = pre_tokenizer
        self.bpe_tokenizer = bpe_tokenizer
        self.bpe_tokenizer_src = bpe_tokenizer_src

        self.idxs, self.probs = idxs, probs

        logger.info(self.__repr__())


    def __getitem__(
        self, index: int
    ) -> Tuple[int, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor],
               Optional[torch.Tensor]]:
        index, source, target, transcript = super().__getitem__(index)

        idxs = get_features_or_waveform(self.idxs[index])
        probs = get_features_or_waveform(self.probs[index])
        idxs = torch.from_numpy(idxs).int()
        probs = torch.from_numpy(probs).float()

        return index, source, target, transcript, idxs, probs

    def collater(self, samples: List[Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor,
                                           torch.Tensor, torch.Tensor]]) -> Dict:
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

        # Add indexes and probabilities of knowledge distillation to the output
        tgt_len = max(idx.shape[0] for _, _, _, _, idx, _ in samples)
        pad_idx = self.tgt_dict.pad()
        idxs, probs = [], []
        for _, _, _, _, idx, prob in samples:
            idxs.append(F.pad(idx, (0, 0, 0, tgt_len - idx.shape[0]), value=pad_idx))
            probs.append(F.pad(prob, (0, 0, 0, tgt_len - prob.shape[0]), value=pad_idx))

        out = {
            "id": indices,
            "net_input": {
                "src_tokens": frames,
                "src_lengths": n_frames,
                "prev_output_tokens": prev_output_tokens,
                "prev_transcript_tokens": prev_transcript_tokens,
            },
            "target": target,
            "target_lengths": target_lengths,
            "transcript": transcript,
            "transcript_lengths": transcript_lengths,
            "ntokens": ntokens,
            "ntokens_transcript": ntokens_transcript,
            "nsentences": len(samples),
            # correct reordering with new batch order
            "teacher_output": [torch.stack(idxs).index_select(0, order),
                               torch.stack(probs).index_select(0, order)],
        }
        return out

class SpeechToTextDatasetCreatorKD(SpeechToTextDatasetCreator):
    # Teacher columns
    KEY_IDX, KEY_PROB = "idx", "probs"

    @classmethod
    def _from_list(
        cls,
        split_name: str,
        is_train_split,
        samples: List[List[Dict]],
        prob_idx: List[List[Dict]],
        data_cfg: S2TDataConfigSrc,
        tgt_dict,
        src_dict,
        pre_tokenizer,
        bpe_tokenizer,
        bpe_tokenizer_src,
    ) -> SpeechToTextDatasetKD:
        audio_paths, n_frames, src_texts, tgt_texts, ids = [], [], [], [], []
        idxs, probs = [], []
        speakers, src_langs, tgt_langs = [], [], []
        for s in samples:
            ids.extend([ss[cls.KEY_ID] for ss in s])
            audio_paths.extend(
                [op.join(data_cfg.audio_root, ss[cls.KEY_AUDIO]) for ss in s]
            )
            n_frames.extend([int(ss[cls.KEY_N_FRAMES]) for ss in s])
            tgt_texts.extend([ss[cls.KEY_TGT_TEXT] for ss in s])
            src_texts.extend(
                [ss.get(cls.KEY_SRC_TEXT, cls.DEFAULT_SRC_TEXT) for ss in s]
            )
            speakers.extend([ss.get(cls.KEY_SPEAKER, cls.DEFAULT_SPEAKER) for ss in s])
            src_langs.extend([ss.get(cls.KEY_SRC_LANG, cls.DEFAULT_LANG) for ss in s])
            tgt_langs.extend([ss.get(cls.KEY_TGT_LANG, cls.DEFAULT_LANG) for ss in s])
        for pi in prob_idx:
            idxs.extend([pii[cls.KEY_IDX] for pii in pi])
            probs.extend([pii[cls.KEY_PROB] for pii in pi])


        return SpeechToTextDatasetKD(
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
            idxs,
            probs,
            tgt_dict,
            src_dict,
            pre_tokenizer,
            bpe_tokenizer,
            bpe_tokenizer_src,
        )

    @classmethod
    def from_tsv(
        cls,
        root: str,
        data_cfg: S2TDataConfigSrc,
        splits: str,
        tgt_dict,
        src_dict,
        pre_tokenizer,
        bpe_tokenizer,
        bpe_tokenizer_src,
        is_train_split: bool,
        epoch: int,
        seed: int,
    ) -> SpeechToTextDatasetKD:
        samples, prob_idx = [], []
        _splits = splits.split(",")
        for split in _splits:
            tsv_path = op.join(root, f"{split}.tsv")
            teacher_path = op.join(root, f"prob_idx_{split}.tsv")
            if not op.isfile(tsv_path):
                raise FileNotFoundError(f"Dataset not found: {tsv_path}")
            if not op.isfile(teacher_path):
                raise FileNotFoundError(f"Teacher dataset not found: {teacher_path}")
            with open(tsv_path) as f:
                reader = csv.DictReader(
                    f,
                    delimiter="\t",
                    quotechar=None,
                    doublequote=False,
                    lineterminator="\n",
                    quoting=csv.QUOTE_NONE,
                )
                samples.append([dict(e) for e in reader])
                assert len(samples) > 0
            with open(teacher_path) as f:
                reader = csv.DictReader(
                    f,
                    delimiter="\t",
                    quotechar=None,
                    doublequote=False,
                    lineterminator="\n",
                    quoting=csv.QUOTE_NONE,
                )
                prob_idx.append([dict(e) for e in reader])
                assert len(prob_idx) > 0

        datasets = [
            cls._from_list(
                name,
                is_train_split,
                [s],
                [p_i],
                data_cfg,
                tgt_dict,
                src_dict,
                pre_tokenizer,
                bpe_tokenizer,
                bpe_tokenizer_src,
            )
            for name, s, p_i in zip(_splits, samples, prob_idx)
        ]

        if is_train_split and len(_splits) > 1 and data_cfg.sampling_alpha != 1.0:
            # temperature-based sampling
            size_ratios = cls._get_size_ratios(
                _splits, [len(s) for s in samples], alpha=data_cfg.sampling_alpha
            )
            datasets = [
                ResamplingDataset(
                    d, size_ratio=r, seed=seed, epoch=epoch, replace=(r >= 1.0)
                )
                for d, r in zip(datasets, size_ratios)
            ]
        return ConcatDataset(datasets)
