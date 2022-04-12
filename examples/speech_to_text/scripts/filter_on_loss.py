#!/usr/bin/env python3 -u
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
import argparse
import logging
import os


def read_losses(loss_f, threshold):
    """
    Reads the file containing the losses generated with the generate_loss.py script
    and identifies the samples to be filterd as those having a loss higher than the
    given threshold.
    """
    filtered_samples = set()
    with open(loss_f, 'r') as fd:
        for line in fd:
            if line.startswith('L-'):
                tokens = line.split('\t')
                if float(tokens[1]) > threshold:
                    filtered_samples.add(int(tokens[0].split('-')[1]))
    return filtered_samples


def filter_data(tsv_in, loss_f, threshold, tsv_out):
    # Setuip logger
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
    )
    logger = logging.getLogger("filter_on_loss")

    filtered_samples = read_losses(loss_f, threshold)
    logger.info("Filtering " + str(len(filtered_samples)) + " samples.")
    with open(tsv_in, 'r') as in_f, open(tsv_out, 'w') as out_f:
        # Copy header
        out_f.write(in_f.readline())
        sample_id = 0
        # Copy only samples that are not filtered
        for line in in_f:
            if sample_id not in filtered_samples:
                out_f.write(line)
            sample_id += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter based on loss")
    parser.add_argument("--tsv-dataset", required=True, help="The tsv containing the dataset definition")
    parser.add_argument("--loss-out", required=True,
                        help="File containing the losses generated with the generate_loss.py script")
    parser.add_argument("--tsv-out", required=True, help="The path where to write the filtered tsv")
    parser.add_argument("--threshold", required=True, type=float,
                        help="The maximum average loss admitted. Everything with a higher loss is filtered.")

    args = parser.parse_args()
    filter_data(args.tsv_dataset, args.loss_out, args.threshold, args.tsv_out)
