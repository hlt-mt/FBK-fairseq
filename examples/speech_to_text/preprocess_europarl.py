#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import shutil
from itertools import groupby
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Tuple

import pandas as pd
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from examples.speech_to_text.data_utils_new import (
    create_zip,
    extract_fbank_features,
    filter_manifest_df,
    gen_config_yaml_with_src,
    gen_vocab,
    get_zip_manifest,
    save_df_to_tsv, asr_normalize,
)

log = logging.getLogger(__name__)


MANIFEST_COLUMNS = ["id", "audio", "n_frames", "src_text", "tgt_text", "speaker"]


class EUROPARL(Dataset):
    SPLITS = ["train", "test", "dev",
              #"train-noisy"
              ]
    LANGUAGES = ["en", "de", "es", "fr", "it", "nl", "pt", "ro", "pl"]

    def __init__(self, root: str, src_lang: str, tgt_lang: str, split: str) -> None:
        assert split in self.SPLITS and src_lang in self.LANGUAGES and tgt_lang in self.LANGUAGES
        _root = Path(root) / f"{src_lang}"
        wav_root, txt_root = _root / "wav", _root / f"{tgt_lang}" / split
        assert _root.is_dir() and wav_root.is_dir() and txt_root.is_dir()
        # Load audio segments
        try:
            import yaml
        except ImportError:
            print("Please install PyYAML to load the MuST-C YAML files")
        with open(_root / f"{tgt_lang}" / f"{split}.yaml") as f:
            segments = yaml.load(f, Loader=yaml.BaseLoader)
        # Load source and target utterances
        for _lang in [src_lang, tgt_lang]:
            with open(txt_root / f"segments.{_lang}") as f:
                utterances = [r.strip() for r in f]
            assert len(segments) == len(utterances)
            for i, u in enumerate(utterances):
                segments[i][_lang] = u
        # Gather info
        self.data = []
        for wav_filename, _seg_group in groupby(segments, lambda x: x["wav"]):
            wav_path = wav_root / wav_filename
            sample_rate = torchaudio.info(wav_path.as_posix())[0].rate
            seg_group = sorted(_seg_group, key=lambda x: x["offset"])
            for i, segment in enumerate(seg_group):
                offset = int(float(segment["offset"]) * sample_rate)
                n_frames = int(float(segment["duration"]) * sample_rate)
                _id = f"{wav_path.stem}_{i}"
                self.data.append(
                    (
                        wav_path.as_posix(),
                        offset,
                        n_frames,
                        sample_rate,
                        segment[src_lang],
                        segment[tgt_lang],
                        segment["speaker_id"],
                        _id,
                    )
                )

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str, str, str]:
        wav_path, offset, n_frames, sr, src_utt, tgt_utt, spk_id, utt_id = self.data[n]
        waveform, _ = torchaudio.load(wav_path, offset=offset, num_frames=n_frames)
        return waveform, sr, src_utt, tgt_utt, spk_id, utt_id

    def __len__(self) -> int:
        return len(self.data)


def process(args):
    root = Path(args.data_root).absolute()
    cur_root = root / f"{args.src_lang}" / f"{args.tgt_lang}"
    if not cur_root.is_dir():
        print(f"{cur_root.as_posix()} does not exist. Skipped.")

    # Extract features
    feature_root = cur_root / "fbank"
    feature_root.mkdir(exist_ok=True)
    for split in EUROPARL.SPLITS:
        print(f"Fetching split {split}...")
        dataset = EUROPARL(root.as_posix(), args.src_lang, args.tgt_lang, split)
        print("Extracting log mel filter bank features...")
        for waveform, sample_rate, _, _, _, utt_id in tqdm(dataset):
            extract_fbank_features(
                waveform, sample_rate, feature_root / f"{utt_id}.npy", args.n_mel_bins
            )
    # Pack features into ZIP
    if args.task == "st":
        zip_path = cur_root / "fbank_st.zip"
    else:
        zip_path = cur_root / "fbank_asr.zip"
    print("ZIPing features...")
    create_zip(feature_root, zip_path)
    print("Fetching ZIP manifest...")
    zip_manifest = get_zip_manifest(zip_path)
    # Generate TSV manifest
    print("Generating manifest...")
    train_text = []
    train_text_src = []
    for split in EUROPARL.SPLITS:
        is_train_split = split.startswith("train")
        manifest = {c: [] for c in MANIFEST_COLUMNS}
        dataset = EUROPARL(root.as_posix(), args.src_lang, args.tgt_lang, split)
        for wav, sr, src_utt, tgt_utt, speaker_id, utt_id in tqdm(dataset):
            manifest["id"].append(utt_id)
            manifest["audio"].append(zip_manifest[utt_id])
            duration_ms = int(wav.size(1) / sr * 1000)
            manifest["n_frames"].append(int(1 + (duration_ms - 25) / 10))
            manifest["src_text"].append(asr_normalize(src_utt) if args.task == "asr" else src_utt)
            manifest["tgt_text"].append(asr_normalize(src_utt) if args.task == "asr" else tgt_utt)
            manifest["speaker"].append(speaker_id)
        if is_train_split:
            train_text.extend(manifest["tgt_text"])
            train_text_src.extend(manifest["src_text"])
        df = pd.DataFrame.from_dict(manifest)
        df = filter_manifest_df(df, is_train_split=is_train_split)
        save_df_to_tsv(df, cur_root / f"{split}_{args.task}_src.tsv")
    # Generate vocab (target)
    v_size_str = "" if args.vocab_type == "char" else str(args.vocab_size)
    if args.vocab_file_tgt == "none":
        spm_filename_prefix = f"spm_{args.vocab_type}{v_size_str}_{args.task}_target"
        with NamedTemporaryFile(mode="w") as f:
            for t in train_text:
                f.write(t + "\n")
            gen_vocab(
                Path(f.name),
                cur_root / spm_filename_prefix,
                args.vocab_type,
                args.vocab_size,
            )
        spm_filename_prefix = spm_filename_prefix + ".model"
    else:
        spm_filename_prefix = args.vocab_file_tgt
    # Generate vocab (source)
    v_size_str = "" if args.vocab_type == "char" else str(args.vocab_size)
    if args.vocab_file_src == "none":
        spm_filename_prefix_src = f"spm_{args.vocab_type}{v_size_str}_{args.task}_source"
        with NamedTemporaryFile(mode="w") as f:
            for t in train_text:
                f.write(t + "\n")
            gen_vocab(
                Path(f.name),
                cur_root / spm_filename_prefix_src,
                args.vocab_type,
                args.vocab_size,
            )
        spm_filename_prefix_src = spm_filename_prefix_src + ".model"
    else:
        spm_filename_prefix_src = args.vocab_file_src
    # Generate config YAML
    gen_config_yaml_with_src(
        cur_root,
        spm_filename_prefix,
        spm_filename_prefix_src,
        yaml_filename=f"config_{args.task}_src.yaml",
        specaugment_policy="ld",
        n_mel_bins=args.n_mel_bins,
    )
    # Clean up
    shutil.rmtree(feature_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-d", required=True, type=str)
    parser.add_argument("--vocab-type", default="unigram", required=True, type=str,
                        choices=["bpe", "unigram", "char"])
    parser.add_argument("--vocab-size", default=8000, type=int)
    parser.add_argument("--task", type=str, choices=["asr", "st"])
    parser.add_argument("--src-lang", type=str, default="en", help="source language")
    parser.add_argument("--tgt-lang", type=str, default="de", help="target language")
    parser.add_argument("--n-mel-bins", default=80, type=int)
    parser.add_argument("--vocab-file-tgt", default="none", type=str,
                        help="absolute path to fairseq target vocabulary file [.txt]")
    parser.add_argument("--vocab-file-src", default="none", type=str,
                        help="absolute path to fairseq source vocabulary file [.txt]")
    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
