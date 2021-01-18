import importlib
import os
from . import *

for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        criterion_name = file[:file.find('.py')]
        importlib.import_module('examples.speech_to_text.criterions.' + criterion_name)