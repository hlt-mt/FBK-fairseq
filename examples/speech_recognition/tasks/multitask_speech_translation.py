# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os

from examples.speech_recognition.data.multitask_dataset import MultiTaskDataset
from examples.speech_recognition.tasks.speech_recognition import SpeechRecognitionTask
from fairseq.data import ConcatDataset
from fairseq.data.data_utils import load_indexed_dataset
from fairseq.tasks import register_task


logger = logging.getLogger(__name__)


@register_task("speech_translation_multitask")
class SpeechTranslationMultiTask(SpeechRecognitionTask):
    """
    Task for training speech recognition model.
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        SpeechRecognitionTask.add_args(parser)
        parser.add_argument('--auxiliary-suffix', default=None, metavar='SUFFIX',
                            help='Suffix to append to target dataset to get the auxiliary targets')

    def __init__(self, args, tgt_dict):
        super().__init__(args, tgt_dict)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        if self.args.dataset_from_json:
            # TODO: not implemented yet
            raise NotImplementedError
        else:
            super().load_dataset(split, combine=combine, **kwargs)
            auxiliary_datasets = []
            for path in self.paths:
                auxiliary_ds = load_indexed_dataset(
                    os.path.join(path, split) + "." + self.args.target_lang + "." + self.args.auxiliary_suffix,
                    None,
                    self.args.dataset_impl)
                if auxiliary_ds is not None:
                    auxiliary_datasets.append(auxiliary_ds)
            assert len(auxiliary_datasets) > 0
            if len(auxiliary_datasets) > 1:
                auxiliary_dataset = ConcatDataset(auxiliary_datasets)
            else:
                auxiliary_dataset = auxiliary_datasets[0]
        assert len(self.datasets[split]) == len(auxiliary_dataset)
        self.datasets[split] = MultiTaskDataset(self.datasets[split], auxiliary_dataset)
