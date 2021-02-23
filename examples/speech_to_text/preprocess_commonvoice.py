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
from typing import Optional, Tuple

import pandas as pd
import torchaudio
from examples.speech_to_text.data_utils_new import (
    create_zip,
    extract_fbank_features,
    filter_manifest_df,
    gen_config_yaml_with_src,
    gen_vocab,
    get_zip_manifest,
    load_df_from_tsv,
    save_df_to_tsv,
    asr_normalize,
)
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm


log = logging.getLogger(__name__)


MANIFEST_COLUMNS = ["id", "audio", "n_frames", "src_text", "tgt_text", "speaker"]


class CommonVoice(Dataset):
    """
    Args:
        root (str): root path to the dataset and generated manifests/features
        source_language (str): source (audio) language
    """

    SPLITS = ["train", "dev", "test"]

    def __init__(
        self,
        root: str,
        split: str,
        source_language: str,
    ) -> None:
        assert split in self.SPLITS
        assert source_language is not None

        self.root: Path = Path(root)

        cv_tsv_path = self.root / f"{split}.tsv"
        assert cv_tsv_path.is_file()

        df = load_df_from_tsv(cv_tsv_path)

        data = df.to_dict(orient="index").items()
        data = [v for k, v in sorted(data, key=lambda x: x[0])]
        self.data = []
        for e in data:
            try:
                path = self.root / "clips" / e["path"]
                _ = torchaudio.info(path.as_posix())
                self.data.append(e)
            except RuntimeError:
                pass

    def __getitem__(
        self, n: int
    ) -> Tuple[Tensor, int, str, str, str]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            tuple: ``(waveform, sample_rate, sentence, speaker_id, sample_id)``
        """
        data = self.data[n]
        path = self.root / "clips" / data["path"]
        waveform, sample_rate = torchaudio.load(path)
        sentence = data["sentence"]
        speaker_id = data["client_id"]
        _id = data["path"].replace(".mp3", "")
        return waveform, sample_rate, sentence, speaker_id, _id

    def __len__(self) -> int:
        return len(self.data)


def process(args):
    root = Path(args.data_root).absolute() / args.src_lang
    if not root.is_dir():
        raise NotADirectoryError(f"{root} does not exist")
    # Extract features
    feature_root = root / "fbank"
    feature_root.mkdir(exist_ok=True)
    for split in CommonVoice.SPLITS:
        print(f"Fetching split {split}...")
        dataset = CommonVoice(root, split, args.src_lang)
        print("Extracting log mel filter bank features...")
        for waveform, sample_rate, _, _, utt_id in tqdm(dataset):
            extract_fbank_features(
                waveform, sample_rate, feature_root / f"{utt_id}.npy", args.n_mel_bins
            )
    # Pack features into ZIP
    zip_path = root / "fbank_asr.zip"
    print("ZIPing features...")
    create_zip(feature_root, zip_path)
    print("Fetching ZIP manifest...")
    zip_manifest = get_zip_manifest(zip_path)
    # Generate TSV manifest
    print("Generating manifest...")
    train_text = []
    train_text_src = []
    task = f"asr_{args.src_lang}"
    for split in CommonVoice.SPLITS:
        manifest = {c: [] for c in MANIFEST_COLUMNS}
        dataset = CommonVoice(root, split, args.src_lang)
        for wav, sr, src_utt, speaker_id, utt_id in tqdm(dataset):
            manifest["id"].append(utt_id)
            manifest["audio"].append(zip_manifest[utt_id])
            duration_ms = int(wav.size(1) / sr * 1000)
            manifest["n_frames"].append(int(1 + (duration_ms - 25) / 10))
            manifest["src_text"].append(asr_normalize(src_utt))
            manifest["tgt_text"].append(asr_normalize(src_utt))
            manifest["speaker"].append(speaker_id)
        is_train_split = split.startswith("train")
        if is_train_split:
            train_text.extend(manifest["tgt_text"])
            train_text_src.extend(manifest["src_text"])
        df = pd.DataFrame.from_dict(manifest)
        df = filter_manifest_df(df, is_train_split=is_train_split)
        save_df_to_tsv(df, root / f"{split}_{task}_src.tsv")
    # Generate vocab
    vocab_size_str = "" if args.vocab_type == "char" else str(args.vocab_size)
    if args.vocab_file_tgt == "none":
        spm_filename_prefix = f"spm_{args.vocab_type}{vocab_size_str}_{task}_target"
        with NamedTemporaryFile(mode="w") as f:
            for t in train_text:
                f.write(t + "\n")
            gen_vocab(
                Path(f.name),
                root / spm_filename_prefix,
                args.vocab_type,
                args.vocab_size
            )
        spm_filename_prefix = spm_filename_prefix + ".model"
    else:
        spm_filename_prefix = args.vocab_file_tgt
    # Generate vocab (source)
    if args.vocab_file_src == "none":
        spm_filename_prefix_src = f"spm_{args.vocab_type}{vocab_size_str}_{task}_source"
        with NamedTemporaryFile(mode="w") as f:
            for t in train_text:
                f.write(t + "\n")
            gen_vocab(
                Path(f.name),
                root / spm_filename_prefix_src,
                args.vocab_type,
                args.vocab_size
            )
        spm_filename_prefix_src = spm_filename_prefix_src + ".model"
    else:
        spm_filename_prefix_src = args.vocab_file_src
    # Generate config YAML
    gen_config_yaml_with_src(
        root,
        spm_filename_prefix,
        spm_filename_prefix_src,
        yaml_filename=f"config_{task}_src.yaml",
        specaugment_policy="ld",
        n_mel_bins=args.n_mel_bins,
    )
    # Clean up
    shutil.rmtree(feature_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", "-d", required=True, type=str,
        help="data root with sub-folders for each language <root>/<src_lang>"
    )
    parser.add_argument(
        "--vocab-type",
        default="unigram",
        required=True,
        type=str,
        choices=["bpe", "unigram", "char"],
    ),
    parser.add_argument("--vocab-size", default=1000, type=int)
    parser.add_argument("--src-lang", "-s", required=True, type=str)
    parser.add_argument("--n-mel-bins", default=80, type=int)
    parser.add_argument("--vocab-file-tgt", default="none", type=str,
                        help="absolute path to fairseq target vocabulary file [.txt]")
    parser.add_argument("--vocab-file-src", default="none", type=str,
                        help="absolute path to fairseq source vocabulary file [.txt]")
    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
