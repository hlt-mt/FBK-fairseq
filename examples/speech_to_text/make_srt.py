# Copyright 2021 FBK

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
from datetime import timedelta

import yaml


def formatted_absolute_time(timestamp, audio_offset):
    """
    Timetamps have to follow the below format:
    hours:minutes:seconds,milliseconds
    where hours, minutes, and seconds are expressed using 2 characters while
    milliseconds are expressed using 3 characters.
    """
    absolute_time = float(timestamp) + audio_offset
    # Replace "." used by datetime library with "," used for milliseconds and add 1 microsecond to ensure time format
    formatted_time = str(timedelta(seconds=absolute_time, microseconds=1)).replace(".", ",")
    # Use "0{HOUR}" instead of only "{HOUR}" in case HOUR < 10, as prescribed by the .srt format
    if formatted_time[:2].endswith(":"):
        formatted_time = "0" + formatted_time
    return formatted_time[:12]  # ignore more than 3 characters for milliseconds


if __name__ == '__main__':
    """
    This script expects 4 positional args:
    - *sub_file*: file containing the subtitles (text with boundaries <eob> and <eol>), one line for each utterance
    - *time_file*: file containing the timestamps per utterance aligned with the sub_file, each segment block has the 
    format: <start_time>:<end_time>, there is a segment block for each <eob>
    - *yaml_file*: file containing the audio offset used to recover the absolute time from the relative time of the 
    utterance
    - *out_path*: folder that will contain the output files in .srt format for each wav
    """
    sub_file = sys.argv[1]
    time_file = sys.argv[2]
    yaml_file = sys.argv[3]
    out_path = sys.argv[4]

    current_wav = ""
    srt_file = None

    with open(yaml_file, 'r') as audios:
        audio_infos = yaml.load(audios, Loader=yaml.BaseLoader)

    with open(sub_file, 'r') as subs, open(time_file, 'r') as times:
        for sub, time, audio in zip(subs, times, audio_infos):
            # Extract the wav name without the extension .wav
            wav_name = audio["wav"][:-4]
            # Open a new file if the previous wav is finished
            if wav_name != current_wav:
                if srt_file is not None:
                    srt_file.close()
                srt_file = open(out_path + "/" + wav_name + ".srt", 'w')
                # Each subtitle block starts with the subtitle id (ascending number starting from 1)
                n_subs = 1
                current_wav = wav_name

            # From each utterance, extract the audio offset, the timestamps and the text for each block
            audio_offset = float(audio["offset"])
            time_list = time.split(" ")
            block_list = sub.split("<eob>")
            for timestamp, block in zip(time_list, block_list):
                start_time, end_time = timestamp.split("-")
                formatted_start_time = formatted_absolute_time(start_time, audio_offset)
                formatted_end_time = formatted_absolute_time(end_time, audio_offset)

                # For each block: the id, the timestamp and the subtitle are printed
                srt_file.write(str(n_subs) + "\n")
                srt_file.write(formatted_start_time + " --> " + formatted_end_time + "\n")
                lines = block.split("<eol>")
                for line in lines:
                    srt_file.write(line.lstrip() + "\n")
                srt_file.write("\n")
                n_subs += 1
        srt_file.close()
