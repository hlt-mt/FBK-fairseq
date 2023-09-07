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

import sys
from examples.speech_to_text.utils.levenshtein_alignment import levenshtein_alignment


def avoid_insertion_at_the_end(distance_string):
    """
    Ensures that the given Levenshtein alignment string ends with the block
    character instead of with insertion/deletions. If not, it fixes the problem.
    """
    if distance_string[-1] == 'B':
        return distance_string
    last_b_idx = distance_string.rfind('B')
    assert all(c == "-" for c in distance_string[last_b_idx + 1:])
    return distance_string[:last_b_idx] + distance_string[last_b_idx + 1:] + 'B'


def generate_target_timestamp(caption, subtitle, timeinterval):
    # Clean text from <eol> and split at <eob>
    caption_cleanl = caption.strip().replace(" <eol>", "").\
        replace("<eol>", "").replace(" <eob>", "<eob>").strip("<eob>").split("<eob>")
    subtitle_cleanl = subtitle.strip().replace(" <eol>", "").\
        replace("<eol>", "").replace(" <eob>", "<eob>").strip("<eob>").split("<eob>")
    # Create text representation for alignment: C stands for character (space included) and B for <eob>
    caption_mask = ["C" * len(sub.strip()) for sub in caption_cleanl]
    subtitle_mask = ["C" * len(sub.strip()) for sub in subtitle_cleanl]
    masked_cap = "B".join(caption_mask) + "B"
    masked_sub = "B".join(subtitle_mask) + "B"
    # List of all the timestamps of the breaks, i.e. the start time
    # of all blocks and the end of the last block
    flatten_timestamps = [start_ts for start_ts, end_ts in timeinterval]
    flatten_timestamps = flatten_timestamps + [timeinterval[-1][-1]]

    # Compute textual alignment between caption and subtitle
    align_cap, align_sub = levenshtein_alignment(masked_cap, masked_sub)

    # Ensure the last character is the block character on both sides.
    # This may not be true if there is an extra block at the end of one side,
    # which is considered a list of insertions after the last block on the other.
    align_cap = avoid_insertion_at_the_end(align_cap)
    align_sub = avoid_insertion_at_the_end(align_sub)

    # Compute timestamp from alignment
    # Set equal initial time
    tgt_timestamp = [flatten_timestamps[0]]
    time_tmp = flatten_timestamps[0]
    flatten_timestamps = flatten_timestamps[1:]
    # Initialize counters at 0
    count = 0
    counts_tmp = []
    for sub_char, cap_char in zip(align_sub, align_cap):
        if sub_char == "C":
            count += 1
        elif cap_char == "-" and sub_char == "B":
            # Save characters number to estimate the timestamp of the unmatched subtitle <eob>
            counts_tmp.append(count)
            count = 0
        elif cap_char == "B":
            # Timestamp estimation from caption based on characters number
            if count != 0 and counts_tmp != []:
                for count_tmp_id in range(len(counts_tmp)):
                    tgt_timestamp.append(
                        (flatten_timestamps[0] - time_tmp) *
                        sum(counts_tmp[:count_tmp_id+1]) / (sum(counts_tmp) + count)
                        + time_tmp)
            if sub_char == "B":
                # Perfect match between caption and subtitle <eob>
                tgt_timestamp.append(flatten_timestamps[0])
            count = 0
            counts_tmp = []
            time_tmp = flatten_timestamps[0]
            flatten_timestamps = flatten_timestamps[1:]

    timestamp = str(tgt_timestamp[0])
    for tgt_time in tgt_timestamp[1:-1]:
        timestamp += f"-{str(tgt_time)} {str(tgt_time)}"
    timestamp += f"-{str(tgt_timestamp[-1])}"
    return timestamp


if __name__ == '__main__':
    """
    This script expects 4 positional args:
    - *caption_file*: file containing the caption (text with boundaries <eob> and <eol>), one line for each utterance
    - *time_file*: file containing the timestamps per utterance aligned with the sub_file, each segment block has the 
    format: <start_time>-<end_time>, there is a segment block for each <eob>
    - *subtitle_file*: file containing the subtitle (text with boundaries <eob> and <eol>), one line for each utterance
    - *out_file*: output file containing the timestamp for the subtitle_file
    """
    cap_file = sys.argv[1]
    time_file = sys.argv[2]
    sub_file = sys.argv[3]
    out_file = sys.argv[4]

    with open(cap_file, 'r') as captions, open(sub_file, 'r') as subtitles, \
            open(time_file, 'r') as times, open(out_file, 'w') as timestamps:
        for caption, subtitle, time in zip(captions, subtitles, times):
            timel = time.strip().split(" ")
            time_intervals = []
            for interval in timel:
                start_time, end_time = interval.split("-")
                time_intervals.append([float(start_time), float(end_time)])
            timestamp = generate_target_timestamp(caption, subtitle, time_intervals)
            timestamps.write(timestamp + "\n")
