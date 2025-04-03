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

import torch
import h5py
from typing import Dict
from examples.speech_to_text.data.occlusion_dataset import OccludedSpeechToTextDataset
from examples.speech_to_text.models.s2t_transformer_fbk import S2TTransformerModel
from examples.speech_to_text.occlusion_explanation.explanation_tasks import ExplanationTask, register_explanation_task
from examples.speech_to_text.occlusion_explanation.perturbators import OcclusionFbankPerturbator
from fairseq.data.concat_dataset import ConcatDataset
from fairseq.data.dictionary import Dictionary


@register_explanation_task("all_tokens")
class AllTokensExplanationTask(ExplanationTask):
    """
    The most generic explanation task. Its goal is to explain the model's prediction for every token 
    in the generated sentence. This is the task introduced in
    `"SPES: Spectrogram Perturbation for Explainable Speech-to-Text Generation
" <https://arxiv.org/abs/2411.01710>`_.
    """

    @staticmethod
    def get_occluded_dataset(
            to_be_occluded_dataset: ConcatDataset,
            perturbator: OcclusionFbankPerturbator,
            tgt_dict: Dictionary) -> OccludedSpeechToTextDataset:
        """
        Args:
            - to_be_occluded_dataset: the dataset to be occluded.
            - perturbator: the perturbator to be used to occlude the data.
            - tgt_dict: the target dictionary.
        Returns:
            Since we are in a generic explanation task, a regular OccludedSpeechToTextDataset object.
        """
        return OccludedSpeechToTextDataset(to_be_occluded_dataset, perturbator, tgt_dict)
    
    @staticmethod
    def save_original_probs(model: S2TTransformerModel, sample: Dict, save_file: str) -> None:
        """
        Performs a forward pass of the model on the sample and saves the output probabilities in a h5 file.
        Args:
            - model: the model to be used to compute the probabilities.
            - sample: the utterances to be used as input to the model, it should
                        contain the tokenized target text on which to perform forced decoding.
            - save_file: the path to the h5 file where to save the probabilities.
        """
        decoder_out, _ = model(**sample["net_input"])
        probs = model.get_normalized_probs(  # (batch_size, padded_seq_len, dict_len)
            decoder_out, log_probs=False)
        assert torch.all((probs >= 0) & (probs <= 1))

        with h5py.File(save_file, "a") as f:
            for i, sample_id in enumerate(sample["id"].tolist()):
                f.create_dataset(
                    str(sample_id),
                    # strip padding
                    data=probs[i, :sample["target_lengths"][i], :].cpu().detach().numpy())
    
    @staticmethod
    def read_original_probs(original_probs_file: h5py.File, sample: Dict) -> Dict:
        """
        Args:
            - original_probs_file: open h5py file containing the original probabilities ouput 
                                   by the model.
            - sample: dictionary containing the sample on which to compute explanations.
        Returns:
            - original_probs: a dictionary containing the original probability distributions output
                              by the model for each utterance in the sample. The keys are the original
                              ids of the utterances and the values are probability distributions tensors
                              of shape (padded_seq_len, dict_len).
        """
        return {
            key: torch.tensor(original_probs_file[str(key)][()])
            for key in torch.unique(sample["orig_id"]).tolist()}

    @staticmethod
    def get_perturbed_probs(model: S2TTransformerModel, sample: Dict) -> Dict:
        """
        Args:
            - model: a model with a decoder of type OcclusionTransformerDecoderScriptable
                     to be used to compute the perturbed probabilities.
            - sample: dictionary containing the sample on which to compute explanations.
        Returns:
            A dictionary containing:
                - perturbed_probs: a tensor containing the perturbed probability distributions output
                               by the model for each utterance in the sample.
                - tgt_embed_masks: a tensor containing the masks indicating where the target
                                embeddings where masked.
        
            The mask for the target embeddings can have various shapes.
            If it is derived from continuous or slic-based perturbations the shape is (batch size, time, channels).
            If it is derived from discrete perturbations the shape is (batch size, sequence length).
        """
        with torch.no_grad():
            decoder_out, ctc_outputs = model(**sample["net_input"])
            tgt_embed_masks = decoder_out[1]["masks"]
            perturb_probs = model.get_normalized_probs(  # (batch_size, padded_seq_len, dict_len)
                decoder_out, log_probs=False)
            assert torch.all((perturb_probs >= 0) & (perturb_probs <= 1))

        return {'perturb_probs': perturb_probs, 'tgt_embed_masks': tgt_embed_masks}
