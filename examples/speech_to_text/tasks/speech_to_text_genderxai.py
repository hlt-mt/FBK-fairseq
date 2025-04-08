# Copyright 2024 FBK

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

from examples.speech_to_text.data.speech_to_text_dataset_genderxai import SpeechToTextDatasetCreatorGenderXai
from fairseq.tasks import register_task
from fairseq.tasks.speech_to_text import SpeechToTextTask


@register_task("speech_to_text_genderxai")
class SpeechToTextGenderXaiTask(SpeechToTextTask):
    
    # The only difference with SpeechToTextTask is in this method, where we use
    # SpeechToTextDatasetCreatorGenderXai instead of SpeechToTextDatasetCreator.
    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        is_train_split = split.startswith("train")
        pre_tokenizer = self.build_tokenizer(self.args)
        bpe_tokenizer = self.build_bpe(self.args)
        self.datasets[split] = SpeechToTextDatasetCreatorGenderXai.from_tsv(
            self.args.data,
            self.data_cfg,
            split,
            self.tgt_dict,
            pre_tokenizer,
            bpe_tokenizer,
            is_train_split=is_train_split,
            epoch=epoch,
            seed=self.args.seed)
