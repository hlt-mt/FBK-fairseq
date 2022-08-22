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

def join_tags_tokens(tags, tokens, dictionary, tags_list):
    """
    Helper function that merges the `token` and `tags` predicted by a model.
    It inserts the tokens corresponding to the predicted `tags` into the predicted `tokens`,
    returning a list of string representation of the tags only and a list of tokens
    that can be converted into a string containing the output of the model with
    the tag predictions inline.
    """
    assert tags.shape[0] == tokens.shape[0]
    tags_strings = []
    joint_string = []
    current_tag = 0
    for t, s in zip(tags, tokens):
        t = t.item()
        if t == 0:
            tags_strings.append("-")
        else:
            tags_strings.append(tags_list[t - 1])
        # join string handling
        if t != current_tag:
            if current_tag != 0:
                joint_string.append(
                    dictionary.index('</{}>'.format(tags_list[current_tag - 1])))
            if t != 0:
                joint_string.append(
                    dictionary.index('<{}>'.format(tags_list[t - 1])))
        joint_string.append(s.item())
        current_tag = t

    return tags_strings, joint_string
