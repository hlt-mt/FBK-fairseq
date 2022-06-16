# Copyright 2021 FBK

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License
import json
from statistics import mean
import torch.nn.functional as F
from torch import FloatTensor, ones_like, arange, cat
import sys
import numbers


def latency_metric(func):
    def prepare_latency_metric(
            delays,
            src_lens,
            ref_lens=None,
            target_padding_mask=None,
    ):
        """
        This script is taken from SimulEval tool.
        delays: bsz, tgt_len
        src_lens: bsz
        target_padding_mask: bsz, tgt_len
        """
        if isinstance(delays, list):
            delays = FloatTensor(delays).unsqueeze(0)

        if len(delays.size()) == 1:
            delays = delays.view(1, -1)

        if isinstance(src_lens, list):
            src_lens = FloatTensor(src_lens)
        if isinstance(src_lens, numbers.Number):
            src_lens = FloatTensor([src_lens])
        if len(src_lens.size()) == 1:
            src_lens = src_lens.view(-1, 1)
        src_lens = src_lens.type_as(delays)

        if ref_lens is not None:
            if isinstance(ref_lens, list):
                ref_lens = FloatTensor(ref_lens)
            if isinstance(ref_lens, numbers.Number):
                ref_lens = FloatTensor([ref_lens])
            if len(ref_lens.size()) == 1:
                ref_lens = ref_lens.view(-1, 1)
            ref_lens = ref_lens.type_as(delays)

        if target_padding_mask is not None:
            tgt_lens = delays.size(-1) - target_padding_mask.sum(dim=1)
            delays = delays.masked_fill(target_padding_mask, 0)
        else:
            tgt_lens = ones_like(src_lens) * delays.size(1)

        tgt_lens = tgt_lens.view(-1, 1)

        return delays, src_lens, tgt_lens, ref_lens, target_padding_mask

    def latency_wrapper(
            delays, src_lens, ref_lens=None, target_padding_mask=None
    ):
        delays, src_lens, tgt_lens, ref_lens, target_padding_mask = prepare_latency_metric(
            delays, src_lens, ref_lens, target_padding_mask)
        return func(delays, src_lens, tgt_lens, ref_lens, target_padding_mask)

    return latency_wrapper


@latency_metric
def length_adaptive_average_lagging(delays, src_lens, tgt_lens, ref_lens=None, target_padding_mask=None):
    """
    Function to calculate Length Adaptive Average Lagging from
    "Over-Generation Cannot Be Rewarded:
    Length-Adaptive Average Lagging for Simultaneous Speech Translation"
    AutoSimTrans Workshop @ NAACL2022

    It modifies the original AL implementation of SimulEval tool
    by taking the maximum between the prediction and the reference length
    in computing the oracle delays to avoid favorable computation for
    under-generative or over-generative models.
    """
    bsz, max_tgt_len = delays.size()

    # Take the maximum between the reference and prediction length to obtain LAAL
    if ref_lens is not None and ref_lens > max_tgt_len:
        max_tgt_len = ref_lens.max().long()
        tgt_lens = ref_lens

    # Only consider the delays that has already larger than src_lens
    lagging_padding_mask = delays >= src_lens
    # Shift left padding to consider one delay that is larger than src_lens
    lagging_padding_mask = F.pad(lagging_padding_mask, (1, 0))[:, :-1]

    if target_padding_mask is not None:
        lagging_padding_mask = lagging_padding_mask.masked_fill(
            target_padding_mask, True)

    # Oracle delays are the delay for the oracle system which goes diagonally
    oracle_delays = (
                        arange(max_tgt_len)
                        .unsqueeze(0)
                        .type_as(delays)
                        .expand([bsz, max_tgt_len])
                    ) * src_lens / tgt_lens

    if delays.size(1) < max_tgt_len:
        oracle_delays = oracle_delays[:, :delays.size(1)]

    if delays.size(1) > max_tgt_len:
        oracle_delays = cat(
            [
                oracle_delays,
                oracle_delays[:, -1]
                * oracle_delays.new_ones(
                    [delays.size(0), delays.size(1) - max_tgt_len]
                )
            ],
            dim=1
        )

    lagging = delays - oracle_delays
    lagging = lagging.masked_fill(lagging_padding_mask, 0)

    # tau is the cut-off step
    tau = (1 - lagging_padding_mask.type_as(lagging)).sum(dim=1)
    al = lagging.sum(dim=1) / tau

    return al


def main(instances_file):
    instances = [json.loads(line) for line in open(instances_file, 'r')]

    laal = []
    laal_ca = []
    for instance in instances:
        delays = instance["delays"]
        elapsed = instance["elapsed"]
        src_lens = instance["source_length"]
        ref_lens = instance["reference_length"]
        laal.append(length_adaptive_average_lagging(delays, src_lens, ref_lens=ref_lens).item())
        laal_ca.append(length_adaptive_average_lagging(elapsed, src_lens, ref_lens=ref_lens).item())

    print(f"LAAL: {round(mean(laal), 3)}")
    print(f"LAAL (CA): {round(mean(laal_ca), 3)}")


if __name__ == "__main__":
    instances_file = sys.argv[1]
    main(instances_file)
