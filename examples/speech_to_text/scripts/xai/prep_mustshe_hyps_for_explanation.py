#!/usr/bin/env python3 -u
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

import argparse
from typing import Dict, List, Tuple
import csv
import pandas as pd
import sentencepiece as spm


OUTPUT_COLUMNS = [
    'id',
    'audio',
    'n_frames',
    'src_text',
    'tgt_text',
    'speaker',
    'found_terms',
    'found_term_pairs',
    'gender_terms_indices',
    'swapped_tgt_text']


def bpe_to_moses(bpe_text: str, sp: spm.SentencePieceProcessor) -> str:
    """
    Get Moses tokenization starting from BPE tokenization.
    Args:
        bpe_text: text tokenized with BPE.
        sp: a sentencepiece model used for BPE tokenization.
    Returns:
        The text tokenized with Moses.
    """
    # Get rid of BPE tokenization
    untokenized_hyp = sp.decode(bpe_text.lower().split())
    # Tokenize with moses
    try:
        from sacremoses import MosesTokenizer
    except ImportError:
        raise ImportError("Please install sacremoses by running 'pip install sacremoses'.")
    mt = MosesTokenizer()
    tokenized_hyp = mt.tokenize(untokenized_hyp, return_str=True)
    return tokenized_hyp


def filter_gender_terms(row: Dict) -> Tuple[str, str]:
    """
    Remove the gender terms which were not generated in the hypothesis.
    Args:
        row: a dictionary representing a MuST-SHE sentence. It should contain 
            - the hypothesis tokenized with moses (field 'moses_tgt_text');
            - the gender terms which should be present according to MuST-SHE annotations 
                (field 'gender_terms').
    Returns:
        - The gender terms present in the hypothesis separated by semicolons.
        - The gender term pairs present in the hypothesis separated by semicolons (the correct term
         followed by the term equivalent in the opposite gender, as annotated in MuST-SHE).
    """
    tokenized_hyp = row['moses_tgt_text'].split()
    # Look for the terms in the lower case, tokenized hypothesis 
    # (without matching the same occurrence twice).
    gender_terms = row['gender_terms'].split(';')
    found_terms = []
    found_term_pairs = []
    for pair in gender_terms:
        correct_term, wrong_term = pair.split(maxsplit=1)
        found_correct, found_wrong = False, False
        if correct_term.lower() in tokenized_hyp:
            pos_found = tokenized_hyp.index(correct_term.lower())
            del tokenized_hyp[pos_found]
            found_correct = True
        if wrong_term.lower() in tokenized_hyp:
            pos_found = tokenized_hyp.index(wrong_term.lower())
            del tokenized_hyp[pos_found]
            found_wrong = True
        # If both the correct and wrong terms are found, we do not include them for our analysis
        # since we cannot determine which one refers to the entity in question.
        if found_correct != found_wrong:  # logical XOR
            if found_correct:
                found_terms.append(correct_term)
            else:
                found_terms.append(wrong_term)
            found_term_pairs.append(correct_term + ' ' + wrong_term)
    return ';'.join(found_terms), ';'.join(found_term_pairs)


def get_bpe_variants(term: str, sp: spm.SentencePieceProcessor) -> List[str]:
    """
    Generate different BPE tokenization variants of the term, including the case where
    the term is capitalized in the reference, but lower case in the hypothesis, or vice versa.
    Args:
        term: The term to be tokenized.
        sp: The sentencepiece model used for BPE tokenization.
    Returns:
        A list of different BPE tokenization variants of the term.
    """
    # Start with the most likely encoding
    term_variants = [' '.join(sp.encode(term, out_type=str)).lower()]
    # Try different capitalizations
    capitalizations = [term, term.lower(), term.capitalize()]
    
    # Generate alternative BPE tokenizations using sampling
    for _ in range(100):
        for term in capitalizations:
            sampled_variant = ' '.join(sp.encode(term, out_type=str, enable_sampling=True, alpha=0.1, nbest_size=-1)).lower()
            if sampled_variant not in term_variants:
                term_variants.append(sampled_variant)
    
    return term_variants


def _find_term_indices(hyp: str, gender_term: str) -> Tuple[str, str]:
    """
    Find the start and end indices of a BPE tokenized gender term in a hypothesis.
    Args:
        hyp: The BPE-tokenized hypothesis.
        gender_term: The BPE-tokenized gender terms to find in the hypothesis.
    Returns:
        - The start and end indices of the gender term in the hypothesis.
        - The hypothesis with the gender term erased to avoid matching it again.
    """
    hyp = hyp.split(' ')
    gender_term = gender_term.split(' ')
    for i, token in enumerate(hyp):
        if token == gender_term[0]:
            j = 1
            while j < len(gender_term) and i + j < len(hyp) and gender_term[j] == hyp[i + j]:
                j += 1 
            if j == len(gender_term):
                for k in range(i, i + j):
                    hyp[k] = '***'
                if gender_term[0] == '▁"':
                    return f"{i + 1}-{i + j - 1}", ' '.join(hyp)
                return f"{i}-{i + j - 1}", ' '.join(hyp)
    raise ValueError(f'Could not find "{gender_term}" in "{hyp}"')


def find_bpe_indices(hypothesis: str, term_bpe: str) -> tuple:
    """
    Find the indices of a BPE tokenized term within the hypothesis, if present.
    Args:
        hypothesis: The BPE-tokenized hypothesis.
        term_bpe: The BPE-tokenized term to find in the hypothesis.
    Returns:
        - The start and end indices of the term in the hypothesis, if found.
        - The hypothesis with the term erased to avoid matching it again.
    """
    # Final spaces are added to avoid matching substrings
    # (e.g. considering '▁el' to be present when the token is '▁ella').
    if term_bpe + ' ' in hypothesis + ' ':
        return _find_term_indices(hypothesis, term_bpe)
    # Theis if is necessary so we are able to find terms that are preceded by quotation marks 
    # in the hypothesis, because then the quotation marks are included in the term's BPE tokenization.
    elif '▁" ' + term_bpe[1:] + ' ' in hypothesis + ' ':
        return _find_term_indices(hypothesis, '▁" ' + term_bpe[1:])
    return None, hypothesis  # No match found


def find_terms_bpe(row: Dict, sp: spm.SentencePieceProcessor) -> str:
    """
    Finds BPE token indices corresponding to gender terms in a tokenized hypothesis.
    Args:
        row: a dictionary containing the fields 'tgt_text' (the BPE-tokenized 
            hypothesis generated by the model) and 'found_terms' (the gender terms
            annotated in MuST-SHE that are present in this hypothesis).
        sp: the sentencepiece model used for BPE tokenization.
    Returns:
        The indices of the BPE tokens in the hypothesis that correspond to the gender terms.
    """
    found_terms = row['found_terms'].split(';')
    hypothesis = row['tgt_text'].lower()
    indices = []

    for term in found_terms:
        for term_bpe in get_bpe_variants(term, sp):
            tok_indices, hypothesis = find_bpe_indices(hypothesis, term_bpe)
            if tok_indices is not None:
                indices.append(tok_indices)
                break  # Stop once a valid match is found

    return ';'.join(indices)


def swap_gender_hypothesis(row: Dict, sp: spm.SentencePieceProcessor) -> str:
    """
    Create a version of the hypothesis where the gender of the annotated term is swapped.
    Args:
        row: a dictionary containing the fields 'tgt_text' (the BPE-tokenized 
            hypothesis generated by the model), 'found_terms' (the gender term present 
            in this hypothesis), 'found_term_pairs' (the pair of correct and wrong terms corresponding
            to the found term, in the MuST-SHE annotation format) and 'gender_terms_indices'  
            (the indices of the BPE tokens in the hypothesis that correspond to the gender terms).
        sp: the sentencepiece model used for BPE tokenization.
    Returns:
        The BPE-tokenized hypothesis with the gender of the gender term swapped.
    """
    hypothesis = row['tgt_text'].split(' ')
    # If there are no gender terms, leave hypothesis unchanged.
    if row['found_terms'] == '' or row['found_term_pairs'] == '' \
            or row['gender_terms_indices'] == '':
        return ' '.join(hypothesis)
    found_terms = row['found_terms'].split(';')
    found_term_pairs = row['found_term_pairs'].split(';')
    gt_indices = row['gender_terms_indices'].split(';')
    for found_term, term_pair, indices in zip(found_terms, found_term_pairs, gt_indices):
        correct_term, wrong_term = term_pair.split(maxsplit=1)
        swap_term = correct_term if found_term == wrong_term else wrong_term
        start, end = indices.split('-')
        start, end = int(start), int(end)
        swap_term = sp.encode(swap_term, out_type=str)
        # Replace the elements in the range with the swapped term.
        hypothesis[start:end + 1] = swap_term 
    return ' '.join(hypothesis)


def main(args: argparse.Namespace):
    """
    Filters the TSV file containing the model's hypotheses for MuST-SHE based on gender coverage.
    Then generates a new version of these hypotheses with swapped gender, so the two versions
    can be compared when generating explanations.
    """
    # Load the hypotheses and MuST-SHE annotations and merge the two dataframes.
    df = pd.read_csv(args.hypotheses_tsv, sep='\t')
    df_mustshe = pd.read_csv(args.mustshe_tsv, sep='\t')
    # ids on both dataframes are slightly different
    df_mustshe['id'] = df_mustshe['ID'] + '_0'
    df = df.merge(df_mustshe[['id', 'GENDERTERMS']], on='id')
    df.rename(columns={'GENDERTERMS': 'gender_terms'}, inplace=True)

    data_list = df.to_dict(orient='records')
    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    for row in data_list:
        # Get moses tokenized hypotheses to look for gender terms.
        row['moses_tgt_text'] = bpe_to_moses(row['tgt_text'], sp)        
        # Filter out gender terms which were not generated in the hypotheses.
        row['found_terms'], row['found_term_pairs'] = filter_gender_terms(row)
    data_list = [row for row in data_list if row['found_terms'] != '']

    # Repeat each row as many times as there are found terms and keep only one gender term per row.
    repeated_data_list = []
    for row in data_list:
        # Find the indices of the remaining gender term tokens in the hypotheses.
        gender_terms_indices = find_terms_bpe(row, sp)
        for i, indices in enumerate(gender_terms_indices.split(';')):
            new_row = row.copy()
            new_row['gender_terms_indices'] = indices
            new_row['found_terms'] = row['found_terms'].split(';')[i]
            new_row['found_term_pairs'] = row['found_term_pairs'].split(';')[i]
            # Create a version of the hypotheses where the gender of the term is swapped.
            new_row['swapped_tgt_text'] = swap_gender_hypothesis(new_row, sp)
            repeated_data_list.append(new_row)

    # Save the resulting tsv.
    df = pd.DataFrame(repeated_data_list)
    df[OUTPUT_COLUMNS].to_csv(args.output_tsv, sep='\t', index=False, quoting=csv.QUOTE_NONE, escapechar='\\')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Prepares a tsv file for explanation generation with hypotheses for MuST-SHE.')
    
    parser.add_argument('--hypotheses-tsv', type=str, required=True,
                        help='Path to a tsv file containing the model\'s hypotheses for MuST-SHE.')
    
    parser.add_argument('--mustshe-tsv', type=str, required=True,
                        help='Path to the original MuST-SHE tsv file with gender annotations.')
    
    parser.add_argument('--spm-model', type=str, required=True,
                        help='Path to the sentencepiece model used for BPE tokenization.')
    
    parser.add_argument('--output-tsv', type=str, required=True,
                        help='Path where to save the new filtered tsv with swapped gender.')
    
    parsed_args = parser.parse_args()
    main(parsed_args)
