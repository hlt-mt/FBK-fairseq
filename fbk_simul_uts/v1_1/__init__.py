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
import importlib
import inspect
import os
import sys
import unittest


for file in os.listdir(os.path.dirname(os.path.dirname(__file__))):
    if file.endswith('.py') and file.startswith('test_'):
        module_name = file[:file.find('.py')]
        if 'fbk_simul_uts.' + module_name not in sys.modules:
            module_dict = importlib.import_module('fbk_simul_uts.' + module_name).__dict__
            for k in module_dict.keys():
                if inspect.isclass(module_dict[k]) and unittest.TestCase in inspect.getmro(module_dict[k]):
                    globals().update({k: module_dict[k]})
