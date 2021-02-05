#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import os.path as op
import shutil
from itertools import groupby
from tempfile import NamedTemporaryFile
from typing import Tuple

import pandas as pd
import torchaudio
from examples.speech_to_text.data_utils import (
    create_zip,
    extract_fbank_features,
    filter_manifest_df,
    gen_config_yaml,
    gen_vocab,
    get_zip_manifest,
    load_df_from_tsv,
    save_df_to_tsv,
)
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from examples.speech_to_text.data_utils_src import gen_config_yaml_with_src

log = logging.getLogger(__name__)


class MUSTC(Dataset):
    """
    Create a Dataset for MuST-C. Each item is a tuple of the form:
    waveform, sample_rate, source utterance, target utterance, speaker_id,
    utterance_id
    """

    SPLITS = ["train", "dev", "tst-COMMON", "tst-HE"]
    LANGUAGES = ["de", "es", "fr", "it", "nl", "pt", "ro", "ru"]

    def __init__(self, root: str, lang: str, split: str) -> None:
        assert split in self.SPLITS and lang in self.LANGUAGES
        _root = op.join(root, f"en-{lang}", "data", split)
        wav_root, txt_root = op.join(_root, "wav"), op.join(_root, "txt")
        assert op.isdir(_root) and op.isdir(wav_root) and op.isdir(txt_root)
        # Load audio segments
        try:
            import yaml
        except ImportError:
            print("Please install PyYAML to load YAML files for " "the MuST-C dataset")
        with open(op.join(txt_root, f"{split}.yaml")) as f:
            segments = yaml.load(f, Loader=yaml.BaseLoader)
        # Load source and target utterances
        for _lang in ["en", lang]:
            with open(op.join(txt_root, f"{split}.{_lang}")) as f:
                utterances = [r.strip() for r in f]
            assert len(segments) == len(utterances)
            for i, u in enumerate(utterances):
                segments[i][_lang] = u
        # Gather info
        self.data = []
        for wav_filename, _seg_group in groupby(segments, lambda x: x["wav"]):
            wav_path = op.join(wav_root, wav_filename)
            sample_rate = torchaudio.info(wav_path)[0].rate
            seg_group = sorted(_seg_group, key=lambda x: x["offset"])
            for i, segment in enumerate(seg_group):
                offset = int(float(segment["offset"]) * sample_rate)
                n_frames = int(float(segment["duration"]) * sample_rate)
                _id = f"{op.splitext(wav_filename)[0]}_{i}"
                self.data.append(
                    (
                        wav_path,
                        offset,
                        n_frames,
                        sample_rate,
                        segment["en"],
                        segment[lang],
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
    if args.task == "st":
        MANIFEST_COLUMNS = ["id", "audio", "n_frames", "src_text", "tgt_text", "speaker"]
    else:
        MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text", "speaker"]
    for lang in MUSTC.LANGUAGES:
        cur_root = op.join(args.data_root, f"en-{lang}")
        if not op.isdir(cur_root):
            print(f"{cur_root} does not exist. Skipped.")
            continue
        # Extract features
        if args.task == "st":
            zip_filename = "fbank80.zip"
            feature_root = op.join(cur_root, "fbank80")
        else:
            zip_filename = "fbank80_asr.zip"
            feature_root = op.join(cur_root, "fbank80_asr")
        zip_path = op.join(cur_root, zip_filename)
        if not os.path.exists(zip_path):
            os.makedirs(feature_root, exist_ok=True)
            for split in MUSTC.SPLITS:
                print(f"Fetching split {split}...")
                dataset = MUSTC(args.data_root, lang, split)
                print("Extracting log mel filter bank features...")
                for waveform, sample_rate, _, _, _, utt_id in tqdm(dataset):
                    extract_fbank_features(
                        waveform, sample_rate, op.join(feature_root, f"{utt_id}.npy")
                    )
            # Pack features into ZIP
            print("ZIPing features...")
            create_zip(feature_root, zip_path)
            # Clean up
            shutil.rmtree(feature_root)

        print("Fetching ZIP manifest...")
        zip_manifest = get_zip_manifest(args.data_root, f"en-{lang}/{zip_filename}")
        # Generate TSV manifest
        print("Generating manifest...")
        train_text = []
        train_text_src = []
        for split in MUSTC.SPLITS:
            is_train_split = split.startswith("train")
            manifest = {c: [] for c in MANIFEST_COLUMNS}
            dataset = MUSTC(args.data_root, lang, split)
            for wav, sr, src_utt, tgt_utt, speaker_id, utt_id in tqdm(dataset):
                manifest["id"].append(utt_id)
                manifest["audio"].append(zip_manifest[utt_id])
                duration_ms = int(wav.size(1) / sr * 1000)
                manifest["n_frames"].append(int(1 + (duration_ms - 25) / 10))
                if args.task == "st":
                    manifest["src_text"].append(src_utt)
                manifest["tgt_text"].append(src_utt if args.task == "asr" else tgt_utt)
                manifest["speaker"].append(speaker_id)
            if is_train_split:
                train_text.extend(manifest["tgt_text"])
                if args.task == "st":
                    train_text_src.extend(manifest["src_text"])
            df = pd.DataFrame.from_dict(manifest)
            df = filter_manifest_df(df, is_train_split=is_train_split)
            save_df_to_tsv(df, op.join(cur_root, f"{split}_{args.task}_src.tsv"))
        # Generate vocab (target)
        v_size_str = "" if args.vocab_type == "char" else str(args.vocab_size)
        spm_filename_prefix = f"spm_{args.vocab_type}{v_size_str}_{args.task}_translation_src"
        with NamedTemporaryFile(mode="w") as f:
            for t in train_text:
                f.write(t + "\n")
            gen_vocab(
                f.name,
                op.join(cur_root, spm_filename_prefix),
                args.vocab_type,
                args.vocab_size,
            )
        # Generate vocab (source) using the same size of target vocab
        spm_filename_prefix_src = f"spm_{args.vocab_type}{v_size_str}_{args.task}_transcription_src"
        with NamedTemporaryFile(mode="w") as f:
            for t in train_text_src:
                f.write(t + "\n")
            gen_vocab(
                f.name,
                op.join(cur_root, spm_filename_prefix_src),
                args.vocab_type,
                args.vocab_size,
            )
        # Generate config YAML
        gen_config_yaml_with_src(
            cur_root,
            spm_filename_prefix + ".model",
            spm_filename_prefix_src + ".model",
            yaml_filename=f"config_{args.task}_src.yaml",
            specaugment_policy="lb",
        )

def process_joint(args):
    # TODO: modify for transcripts
    assert all(
        op.isdir(op.join(args.data_root, f"en-{lang}")) for lang in MUSTC.LANGUAGES
    ), "do not have downloaded data available for all 8 languages"
    cur_root = args.data_root
    # Generate vocab
    vocab_size_str = "" if args.vocab_type == "char" else str(args.vocab_size)
    spm_filename_prefix = f"spm_{args.vocab_type}{vocab_size_str}_{args.task}"
    with NamedTemporaryFile(mode="w") as f:
        for lang in MUSTC.LANGUAGES:
            tsv_path = op.join(cur_root, f"en-{lang}", f"train_{args.task}.tsv")
            df = load_df_from_tsv(tsv_path)
            for t in df["tgt_text"]:
                f.write(t + "\n")
        gen_vocab(
            f.name,
            op.join(cur_root, spm_filename_prefix),
            args.vocab_type,
            args.vocab_size,
        )
    # Generate config YAML
    gen_config_yaml(
        cur_root,
        spm_filename_prefix + ".model",
        yaml_filename=f"config_{args.task}.yaml",
        specaugment_policy="lb",
        prepend_tgt_lang_tag=(args.task == "st"),
    )
    # Make symbolic links to manifests
    for lang in MUSTC.LANGUAGES:
        for split in MUSTC.SPLITS:
            src_path = op.join(cur_root, f"en-{lang}", f"{split}_{args.task}.tsv")
            desc_path = op.join(cur_root, f"{split}_{lang}_{args.task}.tsv")
            if not op.islink(desc_path):
                os.symlink(src_path, desc_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-d", required=True, type=str)
    parser.add_argument(
        "--vocab-type",
        default="unigram",
        required=True,
        type=str,
        choices=["bpe", "unigram", "char"],
    ),
    parser.add_argument("--vocab-size", default=8000, type=int)
    parser.add_argument("--task", type=str, choices=["asr", "st"])
    parser.add_argument("--joint", action="store_true", help="")
    args = parser.parse_args()

    if args.joint:
        process_joint(args)
    else:
        process(args)


if __name__ == "__main__":
    main()
