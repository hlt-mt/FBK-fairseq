#!/usr/bin/env python3
# Copyright (c) FBK.
#
# This source code is licensed under the MIT license.

import argparse
import logging
import shutil
from itertools import groupby
from pathlib import Path
from tempfile import NamedTemporaryFile
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
    gen_config_yaml_with_src,
    gen_vocab,
    get_zip_manifest,
    save_df_to_tsv, asr_normalize, filter_train_manifest_df,
)
from fairseq.data.audio.audio_utils import get_waveform

log = logging.getLogger(__name__)


MANIFEST_COLUMNS = ["id", "audio", "n_frames", "src_text", "tgt_text", "speaker"]


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
        # Load source and target utterances
        for _lang in [src_lang, tgt_lang]:
            with open(txt_root / f"{split}.{_lang}") as f:
                utterances = [r.strip() for r in f]
            assert len(segments) == len(utterances)
            for i, u in enumerate(utterances):
                segments[i][_lang] = u
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
                        segment["speaker_id"],
                        _id,
                    )
                )

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str, str, str]:
        wav_path, offset, n_frames, sr, src_utt, tgt_utt, spk_id, utt_id = self.data[n]
        waveform, _ = get_waveform(wav_path, frames=n_frames, start=offset)
        waveform = torch.from_numpy(waveform)
        return waveform, sr, src_utt, tgt_utt, spk_id, utt_id

    def __len__(self) -> int:
        return len(self.data)


def process(args):
    save_root = Path(args.save_dir).absolute()
    data_dir = Path(args.data_root).absolute()
    if not save_root.is_dir():
        print(f"{data_dir.as_posix()} does not exist. Skipped.")

    # Extract features
    if not args.no_filterbank_extraction:
        feature_root = save_root / "fbank"
        feature_root.mkdir(exist_ok=True)
        for split in args.splits:
            print(f"Fetching split {split}...")
            dataset = YamlDataset(data_dir.as_posix(), args.wav_dir, split, args.src_lang, args.tgt_lang)
            print("Extracting log mel filter bank features...")
            for waveform, sample_rate, _, _, _, utt_id in tqdm(dataset):
                extract_fbank_features(
                    waveform, sample_rate, feature_root / f"{utt_id}.npy", args.n_mel_bins
                )
    # Pack features into ZIP
    zip_path = save_root / "fbank.zip"
    if not args.no_filterbank_extraction:
        print("ZIPing features...")
        create_zip(feature_root, zip_path)
    print("Fetching ZIP manifest...")
    zip_manifest = get_zip_manifest(zip_path)
    # Generate TSV manifest
    print("Generating manifest...")
    train_text = []
    train_text_src = []
    for split in args.splits:
        is_train_split = split.startswith("train")
        manifest = {c: [] for c in MANIFEST_COLUMNS}
        dataset = YamlDataset(data_dir.as_posix(), args.wav_dir, split, args.src_lang, args.tgt_lang)
        for wav, sr, src_utt, tgt_utt, speaker_id, utt_id in tqdm(dataset):
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
            manifest["speaker"].append(speaker_id)
        if is_train_split:
            train_text.extend(manifest["tgt_text"])
            train_text_src.extend(manifest["src_text"])
        df = pd.DataFrame.from_dict(manifest)
        if is_train_split:
            df = filter_train_manifest_df(df)
        save_df_to_tsv(df, save_root / f"{split}_{args.task}_src.tsv")
    # Generate vocab (target)
    v_size_str = "" if args.vocab_type == "char" else str(args.vocab_size)
    if args.vocab_file_tgt == "none":
        spm_filename_prefix = f"spm_{args.vocab_type}{v_size_str}_{args.task}_target"
        with NamedTemporaryFile(mode="w") as f:
            for t in train_text:
                f.write(t + "\n")
            gen_vocab(
                Path(f.name),
                save_root / spm_filename_prefix,
                args.vocab_type,
                args.vocab_size,
            )
        spm_filename_prefix = spm_filename_prefix + ".model"
    else:
        spm_filename_prefix = args.vocab_file_tgt
    # Generate vocab (source)
    if args.task == "st":
        v_size_str = "" if args.vocab_type == "char" else str(args.vocab_size)
        if args.vocab_file_src == "none":
            spm_filename_prefix_src = f"spm_{args.vocab_type}{v_size_str}_{args.task}_source"
            with NamedTemporaryFile(mode="w") as f:
                for t in train_text:
                    f.write(t + "\n")
                gen_vocab(
                    Path(f.name),
                    save_root / spm_filename_prefix_src,
                    args.vocab_type,
                    args.vocab_size,
                )
            spm_filename_prefix_src = spm_filename_prefix_src + ".model"
        else:
            spm_filename_prefix_src = args.vocab_file_src
    else:
        spm_filename_prefix_src = spm_filename_prefix
    # Generate config YAML
    gen_config_yaml_with_src(
        save_root,
        spm_filename_prefix,
        spm_filename_prefix_src,
        yaml_filename=f"config_{args.task}.yaml",
        specaugment_policy="ld",
        n_mel_bins=args.n_mel_bins,
    )
    # Clean up
    if not args.no_filterbank_extraction:
        shutil.rmtree(feature_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-data", required=True, type=str)
    parser.add_argument("--save-dir", "-save", required=True, type=str)
    parser.add_argument("--wav-dir", "-wav", required=True, type=str)
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
    parser.add_argument("--no-filterbank-extraction", action="store_true",
                        help="no mel filterbanks feature extraction")
    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
