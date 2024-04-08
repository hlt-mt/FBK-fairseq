# Copyright 2024 FBK

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

from typing import Dict

import h5py

import torch
from torch import Tensor


def read_feature_attribution_maps_from_h5(explanation_path: str) -> Dict[int, Dict[str, Tensor]]:
    explanations = {}
    with h5py.File(explanation_path, "r") as f:
        for key in f.keys():
            explanations[int(key)] = {}
            group = f[key]
            explanations[int(key)]["fbank_heatmap"] = torch.from_numpy(
                group["fbank_heatmap"][()])
            explanations[int(key)]["tgt_embed_heatmap"] = torch.from_numpy(
                group["tgt_embed_heatmap"][()])
            tgt_txt = group["tgt_text"][()]
            explanations[int(key)]["tgt_text"] = [x.decode('UTF-8') for x in tgt_txt.tolist()]
    return explanations
