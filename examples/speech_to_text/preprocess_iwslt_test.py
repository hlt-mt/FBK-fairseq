#!/usr/bin/env python3
# Copyright (c) FBK.
#
# This source code is licensed under the MIT license.

import argparse
import logging
import shutil
from itertools import groupby
from pathlib import Path
from typing import Tuple

import pandas as pd
import soundfile as sf
import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from examples.speech_to_text.data_utils_new import (
    create_zip,
    extract_fbank_features,
    get_zip_manifest,
    save_df_to_tsv, asr_normalize,
)
from fairseq.data.audio.audio_utils import get_waveform

log = logging.getLogger(__name__)


MANIFEST_COLUMNS = ["id", "audio", "n_frames", "src_text", "tgt_text"]


class YamlDataset(Dataset):
    def __init__(self, root: str, wav_root: str, split: str, src_lang: str, tgt_lang: str) -> None:
        txt_root = Path(root)
        wav_root = Path(wav_root)
        assert wav_root.is_dir() and txt_root.is_dir()
        # Load audio segments
        try:
            import yaml
        except ImportError:
            print("Please install PyYAML to load YAML files")
        with open(txt_root / f"{split}.yaml") as f:
            segments = yaml.load(f, Loader=yaml.BaseLoader)
        # dummy text
        for _lang in [src_lang, tgt_lang]:
            for i, u in enumerate(segments):
                segments[i][_lang] = "dummy"
        # Gather info
        self.data = []
        for wav_filename, _seg_group in groupby(segments, lambda x: x["wav"]):
            wav_path = wav_root / wav_filename
            sample_rate = sf.info(wav_path.as_posix()).samplerate
            seg_group = sorted(_seg_group, key=lambda x: float(x["offset"]))
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
                        _id,
                    )
                )

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str, str]:
        wav_path, offset, n_frames, sr, src_utt, tgt_utt, utt_id = self.data[n]
        waveform, _ = get_waveform(wav_path, frames=n_frames, start=offset)
        waveform = torch.from_numpy(waveform)
        return waveform, sr, src_utt, tgt_utt, utt_id

    def __len__(self) -> int:
        return len(self.data)


def process(args):
    cur_root = Path(args.data_root).absolute()
    if not cur_root.is_dir():
        print(f"{cur_root.as_posix()} does not exist. Skipped.")

    # Extract features
    feature_root = cur_root / "fbank"
    feature_root.mkdir(exist_ok=True)
    for split in args.splits:
        print(f"Fetching split {split}...")
        dataset = YamlDataset(cur_root.as_posix(), args.wav_dir, split, args.src_lang, args.tgt_lang)
        print("Extracting log mel filter bank features...")
        for waveform, sample_rate, _, _, utt_id in tqdm(dataset):
            extract_fbank_features(
                waveform, sample_rate, feature_root / f"{utt_id}.npy", args.n_mel_bins
            )
    # Pack features into ZIP
    zip_path = cur_root / "fbank.zip"
    print("ZIPing features...")
    create_zip(feature_root, zip_path)
    print("Fetching ZIP manifest...")
    zip_manifest = get_zip_manifest(zip_path)
    # Generate TSV manifest
    print("Generating manifest...")
    for split in args.splits:
        manifest = {c: [] for c in MANIFEST_COLUMNS}
        dataset = YamlDataset(cur_root.as_posix(), args.wav_dir, split, args.src_lang, args.tgt_lang)
        for wav, sr, src_utt, tgt_utt, utt_id in tqdm(dataset):
            manifest["id"].append(utt_id)
            manifest["audio"].append(zip_manifest[utt_id])
            duration_ms = int(wav.size(1) / sr * 1000)
            manifest["n_frames"].append(int(1 + (duration_ms - 25) / 10))
            if args.task == "asr":
                manifest["src_text"].append(asr_normalize(src_utt) if args.src_normalize else src_utt)
                manifest["tgt_text"].append(asr_normalize(src_utt) if args.src_normalize else src_utt)
            else:
                manifest["src_text"].append(asr_normalize(src_utt) if args.src_normalize else src_utt)
                manifest["tgt_text"].append(tgt_utt)
        df = pd.DataFrame.from_dict(manifest)
        #df = filter_manifest_df(df, is_train_split=False)
        save_df_to_tsv(df, cur_root / f"{split}_{args.task}_src.tsv")
    # Clean up
    shutil.rmtree(feature_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-d", required=True, type=str)
    parser.add_argument("--wav-dir", "-w", required=True, type=str)
    parser.add_argument("--splits", "-s", nargs='+', required=True, type=str)
    parser.add_argument("--vocab-type", default="unigram", required=True, type=str,
                        choices=["bpe", "unigram", "char"])
    parser.add_argument("--src-normalize", action='store_true', default=False)
    parser.add_argument("--src-lang", type=str, default="en", help="source language")
    parser.add_argument("--tgt-lang", type=str, default="de", help="target language")
    parser.add_argument("--vocab-size", default=8000, type=int)
    parser.add_argument("--task", type=str, choices=["asr", "st"])
    parser.add_argument("--n-mel-bins", default=80, type=int)
    parser.add_argument("--vocab-file-tgt", default="none", type=str,
                        help="absolute path to fairseq target vocabulary file [.txt]")
    parser.add_argument("--vocab-file-src", default="none", type=str,
                        help="absolute path to fairseq source vocabulary file [.txt]")
    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
