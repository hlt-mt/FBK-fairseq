#!/usr/bin/env python3 -u
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
import csv
import logging
import os

TALKID_HEADER = "TALK-ID"
PRONOUN_HEADER = "TED-PRONOUN"


def add_pronoun(tsv_in, must_speakers, tsv_out):
    # Setup logger
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
    )
    logger = logging.getLogger("add_speaker_pronoun_to_tsv")

    logger.info("Reading MuST-Speakers definition")
    pronoun_by_talk = {}
    with open(must_speakers, 'r') as speakers_f:
        speakers_reader = csv.DictReader(
            speakers_f,
            delimiter="\t",
            quotechar=None,
            doublequote=False,
            lineterminator="\n",
            quoting=csv.QUOTE_NONE,
        )
        for line in speakers_reader:
            pronoun = line[PRONOUN_HEADER].strip()
            if pronoun.startswith("Multi-"):
                pronoun = pronoun[6:].capitalize()
            pronoun_by_talk[line[TALKID_HEADER]] = pronoun

    logger.info("Start processing dataset")
    with open(tsv_in, 'r') as in_f, open(tsv_out, 'w') as out_f:
        reader = csv.DictReader(
            in_f,
            delimiter="\t",
            quotechar=None,
            doublequote=False,
            lineterminator="\n",
            quoting=csv.QUOTE_NONE,
        )
        new_fields = reader.fieldnames + ["auxiliary_target"]
        writer = csv.DictWriter(
            out_f,
            fieldnames=new_fields,
            delimiter="\t",
            quotechar=None,
            doublequote=False,
            lineterminator="\n",
            quoting=csv.QUOTE_NONE,
        )
        writer.writeheader()
        filtered = 0
        logged_missing_talks = set()
        for sample in reader:
            talk = sample["speaker"][4:]
            if talk not in pronoun_by_talk:
                if talk not in logged_missing_talks:
                    logger.warning(f"Missing talk {talk} in MuST-Speaker.")
                pronoun = "Na"
                logged_missing_talks.add(talk)
            else:
                pronoun = pronoun_by_talk[talk]
            if pronoun == "Na" or pronoun == "Mix":
                filtered += 1
            elif pronoun == "She" or pronoun == "He" or pronoun == "They":
                sample["auxiliary_target"] = pronoun
                writer.writerow(sample)
            else:
                raise ValueError(f"Unexpected pronoun {pronoun}.")

        logger.info("Filtered rows due to unknown pronoun: " + str(filtered))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add speaker pronoun as auxiliary target to a dataset tsv")
    parser.add_argument("--tsv-dataset", required=True, help="The tsv containing the dataset definition")
    parser.add_argument("--tsv-out", required=True, help="The path where to write the enhanced tsv")
    parser.add_argument("--must-speakers", required=True, help="The tsv of MuST-Speaker.")
    parser.add_argument("--must-speakers-talkid-header", default="TALK-ID")
    parser.add_argument("--must-speakers-pronoun-header", default="TED-PRONOUN")

    args = parser.parse_args()
    TALKID_HEADER = args.must_speakers_talkid_header
    PRONOUN_HEADER = args.must_speakers_pronoun_header
    add_pronoun(args.tsv_dataset, args.must_speakers, args.tsv_out)
