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
import csv
import logging
import os


def filter_data(tsv_in, threshold_min, threshold_max, tsv_out):
    # Setuip logger
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
    )
    logger = logging.getLogger("filter_on_char_ratio")
    filtered = 0
    with open(tsv_in, 'r') as in_f, open(tsv_out, 'w') as out_f:
        # Copy header
        reader = csv.DictReader(
            in_f,
            delimiter="\t",
            quotechar=None,
            doublequote=False,
            lineterminator="\n",
            quoting=csv.QUOTE_NONE,
        )
        writer = csv.DictWriter(
            out_f,
            fieldnames=reader.fieldnames,
            delimiter="\t",
            quotechar=None,
            doublequote=False,
            lineterminator="\n",
            quoting=csv.QUOTE_NONE,
        )
        writer.writeheader()
        for sample in reader:
            filtered += 1
            if len(sample['src_text']) > 0:
                ratio = float(len(sample['tgt_text'])) / len(sample['src_text'])
                if threshold_min < ratio < threshold_max:
                    writer.writerow(sample)
                    filtered -= 1
        logger.info("Filtered rows: " + str(filtered))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter based on loss")
    parser.add_argument("--tsv-dataset", required=True, help="The tsv containing the dataset definition")
    parser.add_argument("--tsv-out", required=True, help="The path where to write the filtered tsv")
    parser.add_argument("--threshold-min", required=True, type=float,
                        help="The minimum ratio admitted. Everything with a lower ratio is filtered.")
    parser.add_argument("--threshold-max", required=True, type=float,
                        help="The maximum ratio admitted. Everything with a higher ration is filtered.")

    args = parser.parse_args()
    filter_data(args.tsv_dataset, args.threshold_min, args.threshold_max, args.tsv_out)
