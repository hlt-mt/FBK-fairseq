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

from typing import Dict
import torch
import h5py
from examples.speech_to_text.data.occlusion_dataset import OccludedSpeechToTextDataset
from examples.speech_to_text.data.occlusion_dataset_genderxai import OccludedSpeechToTextDatasetGenderXai
from examples.speech_to_text.data.occlusion_dataset_with_src_genderxai import OccludedSpeechToTextDatasetWithSrcGenderXai
from examples.speech_to_text.data.speech_to_text_dataset_with_src_genderxai import SpeechToTextDatasetWithSrcGenderXai
from examples.speech_to_text.models.s2t_transformer_fbk import S2TTransformerModel
from examples.speech_to_text.occlusion_explanation.explanation_tasks import ExplanationTask, register_explanation_task
from examples.speech_to_text.occlusion_explanation.perturbators import OcclusionFbankPerturbator
from fairseq.data.concat_dataset import ConcatDataset
from fairseq.data.dictionary import Dictionary 


@register_explanation_task("gender")
class GenderExplanationTask(ExplanationTask):
    """
    The goal of this task is to explain the gender terms the model outputs. It aims to find
    the most relevant features that make the model generate a term in one gender instead of another.
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
            Since here the task is to explain the gender of gender terms, we return an 
            OccludedSpeechToTextDatasetGenderXai object, which contains special fields with 
            gender term annotations.
        """
        dataset_type = type(to_be_occluded_dataset.datasets[0])
        assert all(type(d) == dataset_type for d in to_be_occluded_dataset.datasets), \
            "Datasets of different type are not supported. Dataset types are: " \
            f"{[type(d) == dataset_type for d in to_be_occluded_dataset.datasets]}"
        if dataset_type == SpeechToTextDatasetWithSrcGenderXai:
            occluded_dataset_type = OccludedSpeechToTextDatasetWithSrcGenderXai
        else:
            occluded_dataset_type = OccludedSpeechToTextDatasetGenderXai
        return occluded_dataset_type(
            to_be_occluded_dataset, perturbator, tgt_dict)

    @staticmethod
    def save_original_probs(model: S2TTransformerModel, sample: Dict, save_file: str) -> None:
        """
        Performs a forward pass of the model on the sample and saves the output probabilities in a h5 file.
        Args:
            - model: the model to be used to compute the probabilities.
            - sample: the utterances to be used as input to the model, it should
                        contain the tokenized target text in the originally generated and swapped version
                        so forced decoding can be performed on both.
            - save_file: the path to the h5 file where to save the output probabilities.
        """
               
        # Generated hypothesis
        net_input = sample["net_input"]
        
        decoder_out, _ = model(**net_input)
        if not isinstance(decoder_out, tuple):
            # This step is necessary so we can handle both models with and without ctc
            decoder_out = (decoder_out,)
        probs = model.get_normalized_probs(  # (batch_size, padded_seq_len, dict_len)
            decoder_out, log_probs=False)
        assert torch.all((probs >= 0) & (probs <= 1))

        # Swapped hypothesis
        decoder_out_swapped, _ = model(
            net_input["src_tokens"], net_input["src_lengths"], net_input["swapped_prev_output_tokens"])
        if not isinstance(decoder_out_swapped, tuple):
            # This step is necessary so we can handle both models with and without ctc
            decoder_out_swapped = (decoder_out_swapped,)
        swapped_probs = model.get_normalized_probs(  # (batch_size, padded_seq_len, dict_len)
            decoder_out_swapped, log_probs=False)
        assert torch.all((swapped_probs >= 0) & (swapped_probs <= 1))

        with h5py.File(save_file, "a") as f:
            for i, sample_id in enumerate(sample["id"].tolist()):
                f.create_dataset(
                    str(sample_id),
                    data=probs[i, :sample["target_lengths"][i], :].cpu().detach().numpy())  # strip padding
                f.create_dataset(
                    str(sample_id) + "_swapped",
                    data=swapped_probs[i, :sample["swapped_target_lengths"][i], :].cpu().detach().numpy())
    
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
        # Read the probabilities for the original hypotheses
        original_probs = {
            key: torch.tensor(original_probs_file[str(key)][()])
            for key in torch.unique(sample["orig_id"]).tolist()}
        
        # Read the probabilities for the swapped hypotheses
        original_probs.update({
            str(key) + '_swapped': torch.tensor(original_probs_file[str(key) + "_swapped"][()])
            for key in torch.unique(sample["orig_id"]).tolist()})

        return original_probs

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
                - swapped_perturbed_probs: a tensor containing the perturbed probability distributions output
                                 by the model for the swapped version of each utterance in the sample.
        
            The mask for the target embeddings can have various shapes.
            If it is derived from continuous or slic-based perturbations the shape is (batch size, time, channels).
            If it is derived from discrete perturbations the shape is (batch size, sequence length).
        """
        with torch.no_grad():
            # Forward pass on the original hypotheses
            encoder_out = model.encoder(
                src_tokens=sample["net_input"]["src_tokens"],
                src_lengths=sample["net_input"]["src_lengths"])
            gender_term_pos = [int(range.split('-')[0]) for range in sample["gender_terms_indices"]]
            # Here, we don't pass the occlusion_masks, so they are generated randomly by the perturbator
            decoder_out = model.decoder(
                prev_output_tokens=sample["net_input"]["prev_output_tokens"],
                encoder_out=encoder_out,
                last_tokens_to_perturb=gender_term_pos,
                force_masks_length=True)
            tgt_embed_masks = decoder_out[1]["masks"]
            perturb_probs = model.get_normalized_probs(  # (batch_size, padded_seq_len, dict_len)
                decoder_out, log_probs=False)
            assert torch.all((perturb_probs >= 0) & (perturb_probs <= 1))

            # Forward pass on the swapped hypothesis
            # We want the occlusion masks to be the same for the original and swapped hypotheses,
            # so this time we pass the previously generated occlusion_masks to the perturbator
            swapped_decoder_out = model.decoder(
                prev_output_tokens=sample["net_input"]["swapped_prev_output_tokens"],
                encoder_out=encoder_out,
                occlusion_masks=tgt_embed_masks,
                last_tokens_to_perturb=gender_term_pos,
                force_masks_length=True)
            swapped_perturb_probs = model.get_normalized_probs(  # (batch_size, padded_seq_len, dict_len)
                swapped_decoder_out, log_probs=False)
            assert torch.all((swapped_perturb_probs >= 0) & (swapped_perturb_probs <= 1))

        return {
            'perturb_probs': perturb_probs,
            'tgt_embed_masks': tgt_embed_masks,
            'swapped_perturb_probs': swapped_perturb_probs}
