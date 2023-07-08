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

try:
    import pysrt
except ImportError:
    print("Please install pysrt 'pip install pysrt'")
    raise ImportError
import re
import sys


def add_eol_eob(text):
    """
    Replaces the end of line character with <eol> and appends an <eob>.
    <eol> and <eob> are separated by the rest of the text by a single
    whitespace character. . Eg.:

     >>> add_eol_eob("First line \\r\\n second line.")
     'First line <eol> second line. <eob>'
     >>> add_eol_eob("First line \\n second line.")
     'First line <eol> second line. <eob>'
     >>> add_eol_eob("First line\\nsecond line.  ")
     'First line <eol> second line. <eob>'
     >>> add_eol_eob("First line\\n  ")
     'First line <eob>'
    """
    return re.sub(r"\s*\n\s*", " <eol> ", re.sub(r"\s*$", "", text)).strip() + " <eob>"


def main():
    """
    This script takes as input an srt file and returns its corresponding text
    with subtitle segmentation markers <eob> and <eol>.
    Each subtitle block will be a single line in the output file with the <eob> at the end,
    and each newline inside that block will be substituted by an <eol>.
    """
    srt_path = sys.argv[1]
    subs = pysrt.open(srt_path)

    with open(srt_path + ".blocks", 'w') as fp:
        for sub in subs:
            fp.write("%s\n" % add_eol_eob(sub.text))


if __name__ == "__main__":
    main()
