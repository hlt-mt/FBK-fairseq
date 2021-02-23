#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile

import pandas as pd
from examples.speech_to_text.data_utils_new import (
    create_zip,
    extract_fbank_features,
    gen_vocab,
    get_zip_manifest,
    save_df_to_tsv, gen_config_yaml_with_src, asr_normalize,
)
from torchaudio.datasets import LIBRISPEECH
from tqdm import tqdm


log = logging.getLogger(__name__)

SPLITS = [
    "train-clean-100",
    "train-clean-360",
    #"train-other-500",
    "dev-clean",
    #"dev-other",
    "test-clean",
    #"test-other",
]

MANIFEST_COLUMNS = ["id", "audio", "n_frames", "src_text", "tgt_text", "speaker"]


def process(args):
    out_root = Path(args.output_root).absolute()
    out_root.mkdir(exist_ok=True)
    # Extract features
    feature_root = out_root / "fbank_asr"
    feature_root.mkdir(exist_ok=True)
    for split in SPLITS:
        print(f"Fetching split {split}...")
        dataset = LIBRISPEECH(out_root.as_posix(), url=split, download=True)
        print("Extracting log mel filter bank features...")
        for wav, sample_rate, _, spk_id, chapter_no, utt_no in tqdm(dataset):
            sample_id = f"{spk_id}-{chapter_no}-{utt_no}"
            extract_fbank_features(
                wav, sample_rate, feature_root / f"{sample_id}.npy", args.n_mel_bins
            )
    # Pack features into ZIP
    zip_path = out_root / "fbank_asr.zip"
    print("ZIPing features...")
    create_zip(feature_root, zip_path)
    print("Fetching ZIP manifest...")
    zip_manifest = get_zip_manifest(zip_path)
    # Generate TSV manifest
    print("Generating manifest...")
    train_text = []
    train_text_src = []
    for split in SPLITS:
        manifest = {c: [] for c in MANIFEST_COLUMNS}
        dataset = LIBRISPEECH(out_root.as_posix(), url=split)
        for wav, sample_rate, utt, spk_id, chapter_no, utt_no in tqdm(dataset):
            sample_id = f"{spk_id}-{chapter_no}-{utt_no}"
            manifest["id"].append(sample_id)
            manifest["audio"].append(zip_manifest[sample_id])
            duration_ms = int(wav.size(1) / sample_rate * 1000)
            manifest["n_frames"].append(int(1 + (duration_ms - 25) / 10))
            manifest["src_text"].append(asr_normalize(utt))
            manifest["tgt_text"].append(asr_normalize(utt))
            manifest["speaker"].append(spk_id)
        save_df_to_tsv(
            pd.DataFrame.from_dict(manifest), out_root / f"{split}_src.tsv"
        )
        if split.startswith("train"):
            train_text.extend(manifest["tgt_text"])
            train_text_src.extend(manifest["src_text"])
    # Generate vocab (target)
    vocab_size = "" if args.vocab_type == "char" else str(args.vocab_size)
    if args.vocab_file_tgt == "none":
        spm_filename_prefix = f"spm_{args.vocab_type}{vocab_size}_target"
        with NamedTemporaryFile(mode="w") as f:
            for t in train_text:
                f.write(t + "\n")
            gen_vocab(
                Path(f.name),
                out_root / spm_filename_prefix,
                args.vocab_type,
                args.vocab_size,
            )
        spm_filename_prefix = spm_filename_prefix + ".model"
    else:
        spm_filename_prefix = args.vocab_file_tgt
    # Generate vocab (source)
    spm_filename_prefix_src = f"spm_{args.vocab_type}{vocab_size}_source"
    if args.vocab_file_src == "none":
        with NamedTemporaryFile(mode="w") as f:
            for t in train_text:
                f.write(t + "\n")
            gen_vocab(
                Path(f.name),
                out_root / spm_filename_prefix_src,
                args.vocab_type,
                args.vocab_size,
            )
        spm_filename_prefix_src = spm_filename_prefix_src + ".model"
    else:
        spm_filename_prefix_src = args.vocab_file_src
    # Generate config YAML
    gen_config_yaml_with_src(
        out_root,
        spm_filename_prefix,
        spm_filename_prefix_src,
        yaml_filename=f"config_with_src.yaml",
        specaugment_policy="ld",
        n_mel_bins=args.n_mel_bins,
    )
    # Clean up
    shutil.rmtree(feature_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", "-o", required=True, type=str)
    parser.add_argument(
        "--vocab-type",
        default="unigram",
        required=True,
        type=str,
        choices=["bpe", "unigram", "char"],
    ),
    parser.add_argument("--vocab-size", default=10000, type=int)
    parser.add_argument("--n-mel-bins", default=80, type=int)
    parser.add_argument("--vocab-file-tgt", default="none", type=str,
                        help="absolute path to fairseq target vocabulary file [.txt]")
    parser.add_argument("--vocab-file-src", default="none", type=str,
                        help="absolute path to fairseq source vocabulary file [.txt]")
    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
