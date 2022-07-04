# Copyright 2022 FBK

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

def edit_distance(string1, string2, matrix=None):
    """
    Computes the edit distance using Wagner–Fischer algorithm.
    The matrix is obtained through dynamic programming and contains the Levenshtein distances
    between all prefixes of the first string (string1) and all prefixes of the second (string2).
    The Levenshtein distance between the two full strings corresponds
    to the last value computed (lower right corner).
    """
    if matrix is None:
        matrix = [[0 for _ in range(len(string2) + 1)] for _ in range(len(string1) + 1)]

    for i in range(1, len(string2) + 1):
        matrix[0][i] = i
    for i in range(1, len(string1) + 1):
        matrix[i][0] = i

    for i in range(1, len(string1) + 1):
        for j in range(1, len(string2) + 1):
            delta = 1 if string1[i - 1] != string2[j - 1] else 0
            matrix[i][j] = min(
                matrix[i - 1][j - 1] + delta,
                matrix[i - 1][j] + 1,
                matrix[i][j - 1] + 1
            )
    return matrix[len(string1)][len(string2)], matrix


def optimal_alignment(string1, string2, matrix):
    """
    Returns a string that represents the sequence of the alignment operations need to obtain the optimal alignment.
    Each operation is represented by a different character,
    and the possible operations are only match (M), insertion (I) and deletion (D),
    while substitution is not admitted.
    """
    i, j = len(string1), len(string2)
    operations = ""
    while i > 0 and j > 0:
        diag = matrix[i - 1][j - 1]
        left = matrix[i][j - 1]
        up = matrix[i - 1][j]
        # Precedence: diagonal > left > up
        if diag <= left and diag <= up and string1[i - 1] == string2[j - 1]:
            curr = 'M'
        else:
            curr = 'I' if left <= up else 'D'
        if curr in 'MD':
            i -= 1
        if curr in 'MI':
            j -= 1
        operations = curr + operations

    # edge case: haven't hit (0,0) yet
    operations = 'D' * i + 'I' * j + operations

    return operations


def format_alignment(string1, string2, alignment):
    """
    Takes the two aligned strings string1 and string2,
    and format the alignments operations between the two.
    """
    i, j = 0, 0
    for curr in alignment:
        if curr == 'I':
            string1 = string1[:i] + '-' + string1[i:]
        elif curr == 'D':
            string2 = string2[:j] + '-' + string2[j:]
        i, j = i + 1, j + 1
    return string1, string2


def levenshtein_alignment(string1, string2):
    """
    Compute Levenshtein distance based alignment between string1 and string2.
    """
    edit_dist, matrix = edit_distance(string1, string2)
    alignment = optimal_alignment(string1, string2, matrix)
    align_string1, align_string2 = format_alignment(string1, string2, alignment)
    return align_string1, align_string2
