# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import csv
import logging
import os.path as op
from typing import Dict, List, Optional, Tuple

import torch

from fairseq.data import (
    ConcatDataset,
    Dictionary,
    ResamplingDataset,
    data_utils as fairseq_data_utils,
)
from fairseq.data.audio.feature_transforms import CompositeAudioFeatureTransform
from fairseq.data.audio.speech_to_text_dataset import S2TDataConfig, SpeechToTextDatasetCreator, \
    SpeechToTextDataset, _collate_frames

logger = logging.getLogger(__name__)

class S2TDataConfigSrc(S2TDataConfig):
    """Wrapper class for data config YAML"""
    def __init__(self, yaml_path):
        super().__init__(yaml_path)

    @property
    def vocab_filename_src(self):
        """fairseq source vocabulary file under data root"""
        return self.config.get("vocab_filename_src", "dict.txt")

    @property
    def bpe_tokenizer_src(self) -> Dict:
        """Subword tokenizer to apply after pre-tokenization. Returning
        a dictionary with `bpe` providing the tokenizer name and
        the other items providing the tokenizer-specific arguments.
        Tokenizers are defined in `fairseq.data.encoders.*`"""
        return self.config.get("bpe_tokenizer_src", {"bpe": None})


class SpeechToTextDatasetWithSrc(SpeechToTextDataset):

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
        super().__init__(split, is_train_split, data_cfg, audio_paths, n_frames,
                                            src_texts, tgt_texts, speakers, src_langs, tgt_langs,
                                            ids, tgt_dict, pre_tokenizer, bpe_tokenizer)
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

        logger.info(self.__repr__())

    def tokenize_text_src(self, text: str):
        if self.pre_tokenizer is not None:
            text = self.pre_tokenizer.encode(text)
        if self.bpe_tokenizer_src is not None:
            text = self.bpe_tokenizer_src.encode(text)
        return text

    def __getitem__(
        self, index: int
    ) -> Tuple[int, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        index, source, target = super().__getitem__(index)

        transcript = None
        if self.src_texts is not None:
            tokenized = self.tokenize_text_src(self.src_texts[index])
            transcript = self.src_dict.encode_line(
                tokenized, add_if_not_exist=False, append_eos=True
            ).long()

        return index, source, target, transcript

    def collater(self, samples: List[Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]]) -> Dict:
        #super().collater(samples)
        if len(samples) == 0:
            return {}
        indices = torch.tensor([i for i, _, _, _ in samples], dtype=torch.long)
        frames = _collate_frames(
            [s for _, s, _, _ in samples], self.data_cfg.use_audio_input
        )
        # sort samples by descending number of frames
        n_frames = torch.tensor([s.size(0) for _, s, _, _ in samples], dtype=torch.long)
        n_frames, order = n_frames.sort(descending=True)
        indices = indices.index_select(0, order)
        frames = frames.index_select(0, order)

        target, target_lengths = None, None
        prev_output_tokens = None
        ntokens = None
        if self.tgt_texts is not None:
            target = fairseq_data_utils.collate_tokens(
                [t for _, _, t, _ in samples],
                self.tgt_dict.pad(),
                self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=False,
            )
            target = target.index_select(0, order)
            target_lengths = torch.tensor(
                [t.size(0) for _, _, t, _ in samples], dtype=torch.long
            ).index_select(0, order)
            prev_output_tokens = fairseq_data_utils.collate_tokens(
                [t for _, _, t, _ in samples],
                self.tgt_dict.pad(),
                self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, order)
            ntokens = sum(t.size(0) for _, _, t, _ in samples)

        # Source transcripts
        transcript, transcript_lengths = None, None
        prev_transcript_tokens = None
        ntokens_transcript = None
        if self.src_texts is not None:
            transcript = fairseq_data_utils.collate_tokens(
                [t for _, _, _, t in samples],
                self.src_dict.pad(),
                self.src_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=False,
            )
            transcript = transcript.index_select(0, order)
            transcript_lengths = torch.tensor(
                [t.size(0) for _, _, _, t in samples], dtype=torch.long
            ).index_select(0, order)
            prev_transcript_tokens = fairseq_data_utils.collate_tokens(
                [t for _, _, _, t in samples],
                self.src_dict.pad(),
                self.src_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=True,
            )
            prev_transcript_tokens = prev_transcript_tokens.index_select(0, order)
            ntokens_transcript = sum(t.size(0) for _, _, _, t in samples)

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
        }
        return out

class SpeechToTextDatasetCreatorWithSrc(SpeechToTextDatasetCreator):

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
    ) -> SpeechToTextDatasetWithSrc:
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
        return SpeechToTextDatasetWithSrc(
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
    ) -> SpeechToTextDatasetWithSrc:
        samples = []
        _splits = splits.split(",")
        for split in _splits:
            tsv_path = op.join(root, f"{split}.tsv")
            if not op.isfile(tsv_path):
                raise FileNotFoundError(f"Dataset not found: {tsv_path}")
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

        datasets = [
            cls._from_list(
                name,
                is_train_split,
                [s],
                data_cfg,
                tgt_dict,
                src_dict,
                pre_tokenizer,
                bpe_tokenizer,
                bpe_tokenizer_src
            )
            for name, s in zip(_splits, samples)
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
