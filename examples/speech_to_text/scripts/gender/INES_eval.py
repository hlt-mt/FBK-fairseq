#!/usr/bin/env python3
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
from collections import namedtuple
import os
import logging


InesAccuracy = namedtuple("InesAccuracy", ["term_coverage", "inclusivity_accuracy"])

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),)
LOGGER = logging.getLogger("INES_eval")


def full_match(generated_terms, eval_tokens):
    # Check if the sequence of eval tokens fully matches a subsequence in generated terms
    for i in range(len(generated_terms) - len(eval_tokens) + 1):
        if generated_terms[i:i + len(eval_tokens)] == eval_tokens:
            return True
    return False


def sentence_level_scores(in_f, tsv_f):
    # Calculate sentence-level scores
    sentences = []
    with open(in_f) as i_f, open(tsv_f) as t_f:
        tsv_reader = csv.DictReader(t_f, delimiter='\t')
        for (i_line, terms_f) in zip(i_f, tsv_reader):
            sentence_inclusive = 0
            sentence_not_inclusive = 0
            sentence_found = 0
            generated_terms = i_line.strip().lower().split()
            eval_terms = terms_f['EVAL-TERMS-en'].strip().lower().split(";")
            inclusive_term = eval_terms[0]
            not_inclusive_term = eval_terms[1]

            inclusive_tokens = inclusive_term.split()
            not_inclusive_tokens = not_inclusive_term.split()

            found_inclusive = full_match(generated_terms, inclusive_tokens)
            found_not_inclusive = full_match(generated_terms, not_inclusive_tokens)

            if found_inclusive:
                sentence_inclusive += 1
            if found_not_inclusive:
                sentence_not_inclusive += 1
            if found_inclusive or found_not_inclusive:
                sentence_found += 1
            # check if both inclusive and not_inclusive are matched
            if found_inclusive and found_not_inclusive:
                line_number = terms_f['ID']
                LOGGER.info(f"Both inclusive and not inclusive terms found at line {line_number}: "
                            f"'{i_line.strip()}'")

            sentences.append({
                "num_terms_found": sentence_found,
                "num_inclusive": sentence_inclusive,
                "num_not_inclusive": sentence_not_inclusive
            })

        # asserting both files have been completed
        assert next(i_f, None) is None and next(t_f, None) is None, \
            "INES TSV and hypothesis should have the same length"
    return sentences


def write_sentence_scores(out_f, sentence_scores):
    # Write sentence-level scores to a file
    with open(out_f, 'w') as f_w:
        writer = csv.DictWriter(
            f_w, ["num_terms_found", "num_inclusive", "num_not_inclusive"], delimiter='\t')
        writer.writeheader()
        writer.writerows(sentence_scores)


def global_inclusivity_index(sentence_scores):
    # Calculate global evaluation scores for inclusivity index as % of generated not_inclusive terms
    tot_terms = len(sentence_scores)
    tot_not_inclusive = 0

    for score in sentence_scores:
        tot_not_inclusive += score["num_not_inclusive"]

    if tot_terms == 0:
        raise Exception("Cannot evaluate with empty INES TSV")
    return 1 - float(tot_not_inclusive) / tot_terms


def global_accuracy(sentence_scores):
    # Calculate global evaluation scores for term coverage and inclusivity accuracy
    tot_terms = len(sentence_scores)
    if tot_terms == 0:
        raise Exception("Cannot evaluate with empty INES TSV")
    tot_found = 0
    tot_inclusive = 0
    tot_not_inclusive = 0

    for score in sentence_scores:
        tot_found += score["num_terms_found"]
        tot_inclusive += score["num_inclusive"]
        tot_not_inclusive += score["num_not_inclusive"]

    term_cov = tot_found / tot_terms
    if tot_inclusive + tot_not_inclusive > 0:
        inclusivity_acc = tot_inclusive / (tot_inclusive + tot_not_inclusive)
    else:
        inclusivity_acc = 0.0
    overall_scores = InesAccuracy(term_cov, inclusivity_acc)

    if LOGGER.isEnabledFor(logging.DEBUG):
        LOGGER.debug("Evaluated {} sentences...".format(len(sentence_scores)))
    if LOGGER.isEnabledFor(logging.DEBUG):
        LOGGER.debug("Global: all->{}; found->{}; inclusive->{}; not_inclusive->{}".format(
            tot_terms, tot_found, tot_inclusive, tot_not_inclusive))

    return overall_scores


def print_index_scores(out_scores):
    # Print global evaluation scores for inclusivity index
    print("Global Inclusivity Index")
    print("------------------------")
    print("{}".format(out_scores))


def print_acc_scores(out_scores):
    # Print global evaluation scores
    print("Term Coverage\tInclusivity Accuracy")
    print("-------------------------------------------------")
    print("{}\t{}".format(out_scores.term_coverage, out_scores.inclusivity_accuracy))


if __name__ == '__main__':
    """
    Scripts for the evaluation of gender-inclusive language in MT on INES.
    Given pairs of target inclusive/not-inclusive terms, the evaluation
    scripts calculates:
        - *inclusivity-index*, as the proportion of not_inclusive generated
           by a system. The lower the proportion, the higher the level of
           inclusivity. 
           
    As complementary metrics, given the --acc-scores argument, the scripts 
    can also return:
        - *term coverage*, as the proportion of either inclusive/non-inclusive
          terms generated by a system.
        - *inclusivity accuracy*, as the proportion of desirable inclusive terms
          among all inclusive/not-inclusive terms generated by a system.

    Example usage:
        python INES-eval.py --input your_MT_output.txt --tsv-definition INES.tsv

    Version: 1.0
    """
    print("INES_eval v1.0")
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=str, metavar='FILE',
                        help='Input file to be used to compute scores (it must be tokenized).')
    parser.add_argument('--tsv-definition', required=True, type=str, metavar='FILE',
                        help='TSV INES definitions file.')
    parser.add_argument('--sentence-scores', required=False, default=None, type=str, metavar='FILE',
                        help='If set, sentence level scores are written into this file.')
    parser.add_argument('--acc-scores', required=False, action='store_true', default=False,
                        help='If set, print global accuracy and term coverage.')

    args = parser.parse_args()

    sl_scores = sentence_level_scores(args.input, args.tsv_definition)
    if args.sentence_scores:
        write_sentence_scores(args.sentence_scores, sl_scores)
    scores = global_inclusivity_index(sl_scores)
    print_index_scores(scores)
    if args.acc_scores:
        accuracy_scores = global_accuracy(sl_scores)
        print_acc_scores(accuracy_scores)
