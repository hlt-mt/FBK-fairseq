#!/usr/bin/env python3
# Copyright 2021 FBK

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License
"""
Script to merge MRs with correct formatting.
You need to pass the id of the MR request to merge into master.
Please, make sure that your current repository is in a clean
state before starting the merging process and that you have the .PA_TOKEN
file with your Personal Access token in the git root folder.
Please check gitlab website to know how to create your PA token.
"""
import subprocess
import sys

from pathlib import Path
import requests

PROJECT_BASE_URL = "https://gitlab.com/api/v4/projects/7979857/"

mr_id = sys.argv[1]
git_root = Path(__file__).parent.parent
private_token = (git_root / ".PA_TOKEN").read_text().strip()
mr = requests.get(
    PROJECT_BASE_URL + f"merge_requests/{mr_id}",
    headers={"PRIVATE-TOKEN": private_token}).json()
user_check = input(
    f"[yes/no] Do you want to continue merging MR {mr_id} from {mr['source_branch']} into {mr['target_branch']}?")
if user_check != "yes":
    print("Aborting.")
    sys.exit(1)

# Ensure everything is up-to-date
subprocess.run(["git", "fetch"])
subprocess.run(["git", "checkout", mr['source_branch']])
subprocess.run(["git", "pull"])
subprocess.run(["git", "checkout", mr['target_branch']])
subprocess.run(["git", "pull"])

commit_authors = subprocess.check_output(
    ['git', 'log', f"HEAD..{mr['source_branch']}", '--pretty=format:%an <%ae>']).decode().split('\n')
main_author = sorted(set(commit_authors), key=lambda x: commit_authors.count(x), reverse=True)[0]

subprocess.run(["git", "merge", "--squash", mr['source_branch']])
p = subprocess.Popen(
    ["git", "commit", f'--author="{main_author}"', "-m", f"[{mr['reference']}]{mr['title']}\n\n{mr['description']}"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE)
return_code = p.wait()
assert return_code == 0

subprocess.run(["git", "push"])
return_code = requests.post(
    PROJECT_BASE_URL + f"merge_requests/{mr_id}/notes",
    params={"body": f"Merged to {mr['target_branch']}."},
    headers={"PRIVATE-TOKEN": private_token}).status_code
assert 200 <= return_code < 300
return_code = requests.put(
    PROJECT_BASE_URL + f"merge_requests/{mr_id}",
    params={"state_event": "close"},
    headers={"PRIVATE-TOKEN": private_token}).status_code
assert 200 <= return_code < 300

