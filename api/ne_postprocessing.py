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
import re
import string

PATTERN_TAGS_BEFORE_SPACE = re.compile(r"(<[A-Z_]+>)([\s]+)")


def move_tags_after_space(line):
    """
    Due to the way sentencepiece tokenizes spaces, an opening tag is often inserted
    before a space instead of directly at the beginning of a word.
    This method moves tags that are affected by this problem to the beginning of the following
    word. Eg.:

     >>> move_tags_after_space('We all love<GPE> Italy</GPE>.')
     'We all love <GPE>Italy</GPE>.'
     >>> move_tags_after_space('We all love<WORK_OF_ART> Monna Lisa</WORK_OF_ART>.')
     'We all love <WORK_OF_ART>Monna Lisa</WORK_OF_ART>.'
    """
    for match in PATTERN_TAGS_BEFORE_SPACE.finditer(line):
        line = line[:match.start(0)] + match.group(2) + match.group(1) + line[match.end(0):]
    return line


PATTERN_START_TAG_IN_THE_MIDDLE = re.compile(r"([^\s" + string.punctuation + "]+)(<[A-Z]+>)")
PATTERN_END_TAG_IN_THE_MIDDLE = re.compile(r"(</[A-Z]+>)([^\s" + string.punctuation + "]+)")


def move_tags_to_start_or_end(line):
    """
    Prediciting the tag for each token of a word may lead to situations in which part of
    a word is tagged and part of it is not. In this wrong conditions, this method moves the
    start tags in the middle of words at the beggining of their word, and similarly moves
    the end tags at the end of works.
    Eg.:

     >>> move_tags_to_start_or_end('We all love <GPE>Gree</GPE>ce and Ita<GPE>ly</GPE>.')
     'We all love <GPE>Greece</GPE> and <GPE>Italy</GPE>.'
    """
    for match in PATTERN_START_TAG_IN_THE_MIDDLE.finditer(line):
        line = line[:match.start(0)] + match.group(2) + match.group(1) + line[match.end(0):]
    for match in PATTERN_END_TAG_IN_THE_MIDDLE.finditer(line):
        line = line[:match.start(0)] + match.group(2) + match.group(1) + line[match.end(0):]
    return line
