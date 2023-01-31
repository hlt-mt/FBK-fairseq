# Copyright 2023 FBK
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import argparse
from itertools import groupby
from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from examples.speech_to_text.data_utils import extract_fbank_features
from fairseq.data.audio.audio_utils import get_waveform

try:
    import yaml
except ImportError:
    print("Please install PyYAML to load the YAML files")


class GenericDatasetFromYaml(Dataset):
    def __init__(self, wav_dir: str, yaml_file: str) -> None:
        _wav_dir = Path(wav_dir)
        # Load audio segments
        with open(yaml_file) as f:
            segments = yaml.load(f, Loader=yaml.BaseLoader)
        # Gather info
        self.data = []
        for wav_filename, _seg_group in groupby(segments, lambda x: x["wav"]):
            wav_path = _wav_dir / wav_filename
            sample_rate = sf.info(wav_path.as_posix()).samplerate
            seg_group = sorted(_seg_group, key=lambda x: float(x["offset"]))
            for i, segment in enumerate(seg_group):
                offset = int(float(segment["offset"]) * sample_rate)
                n_frames = int(float(segment["duration"]) * sample_rate)
                self.data.append((
                    wav_path.as_posix(),
                    offset,
                    n_frames,
                    sample_rate))

    def __getitem__(self, n: int) -> Tuple[torch.Tensor, int]:
        wav_path, offset, n_frames, sr = self.data[n]
        waveform, _ = get_waveform(wav_path, frames=n_frames, start=offset)
        waveform = torch.from_numpy(waveform)
        return waveform, sr

    def __len__(self) -> int:
        return len(self.data)


def process(args):
    yaml_paths = args.yaml_path.split(",")
    wav_paths = args.wav_path.split(",")

    sums = 0
    square_sums = 0
    n_samples = 0

    for yaml_path, wav_path in zip(yaml_paths, wav_paths):
        wav_dir = Path(wav_path).absolute()
        yaml_file = Path(yaml_path).absolute()
        assert wav_dir.is_dir()
        # Extract features
        print(f"Fetching data...")
        dataset = GenericDatasetFromYaml(wav_dir.as_posix(), yaml_file.as_posix())
        print("Extracting log mel filter bank features...")
        print("And estimating cepstral mean and variance stats...")
        for waveform, sample_rate in tqdm(dataset):
            features = extract_fbank_features(waveform, sample_rate)
            sums += features.sum(axis=0)
            square_sums += (features ** 2).sum(axis=0)
            n_samples += features.shape[0]

    mean = np.divide(sums, n_samples)
    var = square_sums / n_samples - mean ** 2
    stdev = np.sqrt(np.maximum(var, 1e-8))
    with open(args.save_dir + "/gcmvn.npz", "wb") as f:
        np.savez(f, mean=mean.astype("float32"), std=stdev.astype("float32"))


def main():
    """
    This script computes and stores in numpy format the Global Cepstral Mean and Variance
    features (mean and std deviation) starting from a set of audio files in wav format.
    It takes as input the *wav_path* containing the wav files, the *yaml_path* to the
    yaml file in which the wav files are listed and the *save_dir* in which to save
    the mean and std deviation arrays *gcmvn.npz*.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml-path", "-y", required=True, type=str)
    parser.add_argument("--wav-path", "-w", required=True, type=str)
    parser.add_argument("--save-dir", "-s", required=True, type=str)
    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
