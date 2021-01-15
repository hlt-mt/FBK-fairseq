import importlib
import os

from .s2t_transformer_ctc import *

for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        model_name = file[:file.find('.py')]
        importlib.import_module('examples.speech_to_text.models.' + model_name)