#!/usr/bin/python3
# Copyright 2022 FBK

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
import logging
import os
from typing import List, Dict

MuSTSheEntry = namedtuple("MuSTSheEntry", ["category", "genderterms"])
GenderTerms = namedtuple("GenderTerms", ["correct", "wrong"])
MuSTSheScores = namedtuple("MuSTSheScores", ["term_coverage", "gender_accuracy"])

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
)
LOGGER = logging.getLogger("fairseq_cli.generate")


def read_mustshe(tsv_filename: str) -> List[MuSTSheEntry]:
    """
    Parses MuST-SHE TSV definition file and returns a list of MuSTSheEntry
    that contain the category of each MuST-SHE line (e.g., 1F, 1M, 2F, ...)
    and the list of the genderterms, which contain both the correct and wrong form.
    """
    mustshe_entries = []
    with open(tsv_filename) as t_f:
        tsv_reader = csv.DictReader(t_f, delimiter='\t')
        for line in tsv_reader:
            gender_terms_string = line['GENDERTERMS'].strip().lower().split(";")
            gender_terms = [GenderTerms(*t.split(" ")) for t in gender_terms_string]
            mustshe_entries.append(MuSTSheEntry(line["CATEGORY"], gender_terms))
    return mustshe_entries


def read_predictions(input_filename: str) -> List[str]:
    """
    Reads the predictions of the system and lower-cases them.
    """
    with open(input_filename) as i_f:
        return [line.strip().lower() for line in i_f]


def _find_term(generated_terms: List[str], term: str) -> bool:
    """
    Returns true if term is present in the list of generated terms
    and removes the matched term from the list to avoid re-matching
    the same generated term two times
    """
    try:
        pos_found = generated_terms.index(term)
        del generated_terms[pos_found]
        return True
    except ValueError:
        return False


def sentence_level_statistics(
        predicted_lines: List[str],
        mustshe_definition: List[MuSTSheEntry]) -> List[Dict[str, int]]:
    """
    Returns sentence-level statistics about the gender terms defined
    in the MuST-SHE definition for the predictions in input.
    """
    sentences = []
    for pred_line, mustshe_entry in zip(predicted_lines, mustshe_definition):
        sentence_correct = 0
        sentence_wrong = 0
        sentence_found = 0
        generated_terms = pred_line.split()
        for t in mustshe_entry.genderterms:
            found = False
            if _find_term(generated_terms, t.correct):
                sentence_correct += 1
                found = True
            if _find_term(generated_terms, t.wrong):
                sentence_wrong += 1
                found = True
            if found:
                sentence_found += 1
        sentences.append({
            "num_terms": len(mustshe_entry.genderterms),
            "num_terms_found": sentence_found,
            "num_correct": sentence_correct,
            "num_wrong": sentence_wrong})

    return sentences


def write_sentence_acc(out_f: str, sentence_scores: List[Dict[str, int]]):
    with open(out_f, 'w') as f_w:
        writer = csv.DictWriter(
            f_w, ["num_terms", "num_terms_found", "num_correct", "num_wrong"], delimiter='\t')
        writer.writeheader()
        writer.writerows(sentence_scores)


def global_scores(
        sentence_scores: List[Dict[str, int]],
        mustshe_definition: List[MuSTSheEntry]) -> Dict[str, MuSTSheScores]:
    """
    Computes category-level (and global) scores from sentence level statistics, and
    returns a dictionary where each key represents the category and the values are
    the term coverage and gender accuracy (see https://aclanthology.org/2020.coling-main.350/
    for the description of term coverage and gender accuracy).
    """
    i = 0
    category_buffers = {}
    for mustshe_entry in mustshe_definition:
        if mustshe_entry.category not in category_buffers:
            category_buffers[mustshe_entry.category] = {
                "num_terms": 0, "num_correct": 0, "num_wrong": 0, "num_terms_found": 0}
        for key in ["num_terms", "num_correct", "num_wrong", "num_terms_found"]:
            category_buffers[mustshe_entry.category][key] += sentence_scores[i][key]
        i += 1
    if LOGGER.isEnabledFor(logging.DEBUG):
        LOGGER.debug("Evaluated {} sentences...".format(i))
    overall_scores = {}
    tot_terms = 0
    tot_found = 0
    tot_correct = 0
    tot_wrong = 0
    for c in category_buffers:
        term_cov = float(category_buffers[c]["num_terms_found"]) / category_buffers[c]["num_terms"]
        if category_buffers[c]["num_terms_found"] > 0:
            gender_acc = float(category_buffers[c]["num_correct"]) / (
                category_buffers[c]["num_correct"] + category_buffers[c]["num_wrong"])
        else:
            gender_acc = 0.0
        overall_scores[c] = MuSTSheScores(term_cov, gender_acc)
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug(
                f"Category {c}: all->{category_buffers[c]['num_terms']},"
                f" found->{category_buffers[c]['num_terms_found']};"
                f" correct->{category_buffers[c]['num_correct']};"
                f" wrong->{category_buffers[c]['num_wrong']}")
        tot_terms += category_buffers[c]["num_terms"]
        tot_found += category_buffers[c]["num_terms_found"]
        tot_correct += category_buffers[c]["num_correct"]
        tot_wrong += category_buffers[c]["num_wrong"]
    if LOGGER.isEnabledFor(logging.DEBUG):
        LOGGER.debug(
            f"Global: all->{tot_terms}; found->{tot_found}; "
            f"correct->{tot_correct}; wrong->{tot_wrong}")
    if tot_terms > 0:
        overall_scores["Global"] = MuSTSheScores(
            tot_found / tot_terms, tot_correct / (tot_correct + tot_wrong))
    return overall_scores


def print_scores(category_scores: Dict[str, MuSTSheScores], print_latex: bool = False):
    """
    Prints the category-level (and global scores) in TSV-like and
    (if specified) in LaTeX-friendly format.
    """
    categories = list(category_scores.keys())
    categories.sort()
    print("Category\tTerm Coverage\tGender Accuracy")
    print("-------------------------------------------------")
    for c in categories:
        if c == "Global":
            print("-------------------------------------------------")
        print(f"{c}\t{category_scores[c].term_coverage}\t{category_scores[c].gender_accuracy}")
        if c == "Global":
            print("-------------------------------------------------")
    if print_latex:
        import pandas as pd
        df = pd.DataFrame.from_dict({
            s: category_scores[s]._asdict() for s in category_scores}, orient='index')
        print(df.to_latex())


if __name__ == '__main__':
    """
    Scripts for the evaluation of term coverage and gender accuracy on MuST-SHE.
    A complete description about the two metrics can be found at 
    https://aclanthology.org/2020.coling-main.350.pdf.
    If using, please consider citing:
    - M. Gaido, B. Savoldi et al., "Breeding Gender-aware Direct Speech Translation Systems", COLING 2020,
    Bibtex available at: https://aclanthology.org/2020.coling-main.350/
    
    Version: 1.1
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=str, metavar='FILE',
                        help='Input file to be used to compute accuracies (it must be tokenized).')
    parser.add_argument('--tsv-definition', required=True, type=str, metavar='FILE',
                        help='TSV MuST-SHE definitions file.')
    parser.add_argument('--sentence-acc', required=False, default=None, type=str, metavar='FILE',
                        help='If set, sentence level accuracies are written into this file.')
    parser.add_argument("--print-latex", required=False, action='store_true', default=False)

    args = parser.parse_args()

    mustshe = read_mustshe(args.tsv_definition)
    predictions = read_predictions(args.input)
    # Sanity check that the predictions have the same length of MuST-SHE
    assert len(predictions) == len(mustshe)
    sl_scores = sentence_level_statistics(predictions, mustshe)
    if args.sentence_acc:
        write_sentence_acc(args.sentence_acc, sl_scores)
    scores = global_scores(sl_scores, mustshe)
    print_scores(scores, args.print_latex)
