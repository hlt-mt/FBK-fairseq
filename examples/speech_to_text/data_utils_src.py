#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import csv
import os
import os.path as op
import zipfile
from functools import reduce
from glob import glob
from multiprocessing import cpu_count
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import sentencepiece as sp
from fairseq.data.audio.audio_utils import _get_kaldi_fbank, _get_torchaudio_fbank
from fairseq.data.audio.feature_transforms.utterance_cmvn import UtteranceCMVN
from tqdm import tqdm

def gen_config_yaml_with_src(
    data_root,
    spm_filename,
    spm_filename_src,
    yaml_filename="config.yaml",
    specaugment_policy="lb",
    prepend_tgt_lang_tag=False,
    sampling_alpha=1.0,
):
    data_root = op.abspath(data_root)
    writer = S2TDataConfigWriter(op.join(data_root, yaml_filename))
    writer.set_audio_root(op.abspath(data_root))
    writer.set_vocab_filename(spm_filename.replace(".model", ".txt"))
    writer.set_vocab_filename_src(spm_filename_src.replace(".model", ".txt"))
    writer.set_input_channels(1)
    writer.set_input_feat_per_channel(80)
    specaugment_setters = {
        "lb": writer.set_specaugment_lb_policy,
        "ld": writer.set_specaugment_ld_policy,
        "sm": writer.set_specaugment_sm_policy,
        "ss": writer.set_specaugment_ss_policy,
    }
    assert specaugment_policy in specaugment_setters
    specaugment_setters[specaugment_policy]()
    writer.set_bpe_tokenizer(
        {
            "bpe": "sentencepiece",
            "sentencepiece_model": op.join(data_root, spm_filename),
            "sentencepiece_model_src": op.join(data_root, spm_filename_src),
        }
    )
    if prepend_tgt_lang_tag:
        writer.set_prepend_tgt_lang_tag(True)
    writer.set_sampling_alpha(sampling_alpha)
    writer.set_feature_transforms("_train", ["specaugment"])
    writer.flush()


class S2TDataConfigWriter(object):
    DEFAULT_VOCAB_FILENAME = "dict.txt"
    DEFAULT_INPUT_FEAT_PER_CHANNEL = 80
    DEFAULT_INPUT_CHANNELS = 1

    def __init__(self, yaml_path):
        try:
            import yaml
        except ImportError:
            print("Please install PyYAML to load YAML files for S2T data config")
        self.yaml = yaml
        self.yaml_path = yaml_path
        self.config = {}

    def flush(self):
        with open(self.yaml_path, "w") as f:
            self.yaml.dump(self.config, f)

    def set_audio_root(self, audio_root=""):
        self.config["audio_root"] = audio_root

    def set_vocab_filename(self, vocab_filename="dict.txt"):
        self.config["vocab_filename"] = vocab_filename

    def set_vocab_filename_src(self, vocab_filename_src="dict_src.txt"):
        self.config["vocab_filename_src"] = vocab_filename_src

    def set_specaugment(
        self,
        time_wrap_w: int,
        freq_mask_n: int,
        freq_mask_f: int,
        time_mask_n: int,
        time_mask_t: int,
        time_mask_p: float,
    ):
        self.config["specaugment"] = {
            "time_wrap_W": time_wrap_w,
            "freq_mask_N": freq_mask_n,
            "freq_mask_F": freq_mask_f,
            "time_mask_N": time_mask_n,
            "time_mask_T": time_mask_t,
            "time_mask_p": time_mask_p,
        }

    def set_specaugment_lb_policy(self):
        self.set_specaugment(
            time_wrap_w=0,
            freq_mask_n=1,
            freq_mask_f=27,
            time_mask_n=1,
            time_mask_t=100,
            time_mask_p=1.0,
        )

    def set_specaugment_ld_policy(self):
        self.set_specaugment(
            time_wrap_w=0,
            freq_mask_n=2,
            freq_mask_f=27,
            time_mask_n=2,
            time_mask_t=100,
            time_mask_p=1.0,
        )

    def set_specaugment_sm_policy(self):
        self.set_specaugment(
            time_wrap_w=0,
            freq_mask_n=2,
            freq_mask_f=15,
            time_mask_n=2,
            time_mask_t=70,
            time_mask_p=0.2,
        )

    def set_specaugment_ss_policy(self):
        self.set_specaugment(
            time_wrap_w=0,
            freq_mask_n=2,
            freq_mask_f=27,
            time_mask_n=2,
            time_mask_t=70,
            time_mask_p=0.2,
        )

    def set_input_channels(self, input_channels=1):
        self.config["input_channels"] = input_channels

    def set_input_feat_per_channel(self, input_feat_per_channel=80):
        self.config["input_feat_per_channel"] = input_feat_per_channel

    def set_bpe_tokenizer(self, bpe_tokenizer: Dict[str, Any]):
        self.config["bpe_tokenizer"] = bpe_tokenizer

    def set_feature_transforms(self, split, transforms: List[str]):
        if "transforms" not in self.config:
            self.config["transforms"] = {}
        self.config["transforms"][split] = transforms

    def set_prepend_tgt_lang_tag(self, flag=True):
        self.config["prepend_tgt_lang_tag"] = flag

    def set_sampling_alpha(self, sampling_alpha=1.0):
        self.config["sampling_alpha"] = sampling_alpha
