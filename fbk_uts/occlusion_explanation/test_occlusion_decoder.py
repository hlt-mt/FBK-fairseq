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

import argparse
import unittest

import torch

from examples.speech_to_text.occlusion_explanation.occlusion_transformer_decoder import \
    OcclusionTransformerDecoderScriptable
from examples.speech_to_text.occlusion_explanation.perturbators.tokens import (
    OcclusionDecoderEmbeddingsPerturbatorContinuous,
    OcclusionDecoderEmbeddingsPerturbatorDiscrete)
from fairseq.models.transformer import base_architecture
from fbk_uts.occlusion_explanation.test_occlusion_dataset import MockDictionary


class TestOcclusionDecoderEmbeddingsPerturbator(unittest.TestCase):
    def setUp(self):
        self.embeddings = torch.randn(3, 5, 10)  # Batch size, Sequence length, Embedding dimension
        self.original_embeddings = self.embeddings.clone()

    def test_continuous_perturbator(self):
        torch.manual_seed(0)
        perturbator = OcclusionDecoderEmbeddingsPerturbatorContinuous(
            no_position_occlusion=False, p=0.5)
        perturbed_embeddings, mask = perturbator(self.embeddings)
        self.assertEqual(perturbed_embeddings.shape, (3, 5, 10))
        self.assertEqual(mask.shape, (3, 5, 10))
        expected_mask = torch.tensor([[[0., 1., 0., 0., 0., 1., 0., 1., 0., 1.],
                                       [0., 0., 0., 0., 0., 1., 1., 1., 0., 0.],
                                       [1., 1., 0., 1., 0., 1., 1., 0., 0., 0.],
                                       [0., 1., 0., 0., 0., 0., 0., 1., 1., 1.],
                                       [1., 0., 1., 0., 0., 0., 1., 1., 0., 0.]],
                                      [[1., 1., 1., 1., 1., 0., 1., 0., 0., 0.],
                                       [1., 1., 0., 1., 1., 0., 1., 1., 1., 1.],
                                       [0., 0., 1., 1., 1., 1., 1., 0., 1., 0.],
                                       [0., 1., 1., 0., 1., 0., 0., 0., 1., 0.],
                                       [1., 1., 1., 1., 0., 0., 1., 1., 0., 0.]],
                                      [[1., 1., 0., 0., 0., 0., 0., 0., 0., 1.],
                                       [0., 1., 0., 1., 1., 0., 0., 1., 1., 1.],
                                       [1., 1., 0., 1., 0., 0., 0., 0., 0., 0.],
                                       [0., 1., 0., 0., 1., 0., 1., 0., 1., 1.],
                                       [1., 1., 1., 0., 0., 0., 1., 0., 0., 0.]]])
        self.assertFalse(torch.all(torch.eq(perturbed_embeddings, self.original_embeddings)))
        self.assertTrue(torch.all(torch.eq(mask, expected_mask)))

        # Check that if a mask is provided, it is used instead of generating a new one
        torch.manual_seed(1)
        self.embeddings = self.original_embeddings.clone()
        perturbed_embeddings, mask = perturbator(self.embeddings, occlusion_masks=expected_mask)
        self.assertFalse(torch.all(torch.eq(perturbed_embeddings, self.original_embeddings)))
        self.assertTrue(torch.all(torch.eq(mask, expected_mask)))

        # Check that if last_tokens_to_perturb is provided, no occlusion happens after those positions
        self.embeddings = self.original_embeddings.clone()
        perturbed_embeddings, mask = perturbator(self.embeddings, last_tokens_to_perturb=[3, 3, 3])
        self.assertTrue(torch.all(mask[:, 4:]))
        self.assertTrue(torch.all(torch.eq(perturbed_embeddings[:, 4:], self.original_embeddings[:, 4:])))

    def test_discrete_perturbator(self):
        torch.manual_seed(0)
        perturbator = OcclusionDecoderEmbeddingsPerturbatorDiscrete(
            no_position_occlusion=False, p=0.5)
        self.embeddings = self.original_embeddings.clone()
        perturbed_embeddings, masks = perturbator(self.embeddings)
        expected_masks = torch.tensor([[0., 1., 0., 0., 0.],
                                       [1., 0., 1., 0., 1.],
                                       [0., 0., 0., 0., 0.]])
        self.assertEqual(perturbed_embeddings.shape, (3, 5, 10))
        self.assertEqual(masks.shape, (3, 5))
        self.assertFalse(torch.all(torch.eq(perturbed_embeddings, self.original_embeddings)))
        self.assertTrue(torch.equal(masks, expected_masks))
        for i in range(masks.shape[0]):
            for k in range(masks[i].shape[0]):
                if masks[i][k] == 0:
                    self.assertTrue((perturbed_embeddings[i][k] == 0).all())
                else:
                    self.assertTrue((perturbed_embeddings[i][k] != 0).all())

        # Check that if a mask is provided, it is used instead of generating a new one
        torch.manual_seed(1)
        self.embeddings = self.original_embeddings.clone()
        perturbed_embeddings, masks = perturbator(self.embeddings, occlusion_masks=expected_masks)
        self.assertFalse(torch.all(torch.eq(perturbed_embeddings, self.original_embeddings)))
        self.assertTrue(torch.equal(masks, expected_masks))

        # Check that if last_tokens_to_perturb is provided, no occlusion happens after those positions
        self.embeddings = self.original_embeddings.clone()
        perturbed_embeddings, masks = perturbator(self.embeddings, last_tokens_to_perturb=[3, 3, 3])
        self.assertTrue(torch.all(masks[:, 4:]))
        self.assertTrue(torch.all(torch.eq(perturbed_embeddings[:, 4:], self.original_embeddings[:, 4:])))

    def test_from_config_dict_no_config(self):
        perturbator = OcclusionDecoderEmbeddingsPerturbatorContinuous.from_config_dict(
            config=None)
        self.assertEqual(perturbator.no_position_occlusion, True)
        self.assertEqual(perturbator.p, 0.5)

    def test_from_config_dict(self):
        mock_config = {"decoder_occlusion": {"no_position_occlusion": False, "p": 0.2}}
        perturbator = OcclusionDecoderEmbeddingsPerturbatorContinuous.from_config_dict(
            config=mock_config)
        self.assertEqual(perturbator.no_position_occlusion, False)
        self.assertEqual(perturbator.p, 0.2)


class TestOcclusionTransformerDecoderScriptable(unittest.TestCase):
    def setUp(self):
        self.args = argparse.Namespace()
        base_architecture(self.args)
        self.args.max_target_positions = 1024
        self.dictionary = MockDictionary()
        self.perturbator = OcclusionDecoderEmbeddingsPerturbatorContinuous()
        self.decoder = OcclusionTransformerDecoderScriptable(
            self.args, self.dictionary, self.perturbator)
        self.prev_output_tokens = torch.randint(8, size=(3, 7))
        self.encoder_out = {
            "encoder_out": [torch.randn(5, 3, 512)],  # T x B x C
            "encoder_padding_mask": [torch.zeros(3, 5).bool()],  # B x T
            "encoder_embedding": [],
            "encoder_states": [],
            "src_tokens": [],
            "src_lengths": []}

    # check whether the sizes are as expected
    def test_embed_tokens_positions(self):
        positions = torch.randn(3, 7, 512)
        x, masks = self.decoder.embed_tokens_positions(self.prev_output_tokens, positions)
        self.assertEqual(x.size(), torch.Size([3, 7, 512]))
        self.assertEqual(masks.size(), torch.Size([3, 7, 512]))

    # check whether the occlusion of position works as expected
    def test_embed_tokens_with_without_positions(self):
        positions = torch.ones(3, 7, 512)
        self.perturbator.no_position_occlusion = True
        x, masks = self.decoder.embed_tokens_positions(self.prev_output_tokens, positions)
        # all elements are non-zero since they are summed to the positions
        self.assertTrue(torch.count_nonzero(x).item() == 10752)
        self.perturbator.no_position_occlusion = False
        x, masks = self.decoder.embed_tokens_positions(self.prev_output_tokens, positions)
        # some elements might be zero since also the positions are occluded
        self.assertTrue(torch.count_nonzero(x).item() < 10752)

    def test_extract_features_scriptable(self):
        output, _ = self.decoder.extract_features_scriptable(
            self.prev_output_tokens, self.encoder_out)
        self.assertEqual(output.shape, torch.Size([3, 7, 512]))  # batch, tgt_len, decoder_embed_dim

    # Check that extract_features() also returns model_specific_outputs
    def test_extract_features(self):
        output, model_specific_output = self.decoder.extract_features(
            self.prev_output_tokens, self.encoder_out)
        self.assertEqual(output.shape, torch.Size([3, 7, 512]))  # batch, tgt_len, decoder_embed_dim
        self.assertTrue("masks" in model_specific_output.keys())

    def test_forward(self):
        output, _ = self.decoder.forward(self.prev_output_tokens, self.encoder_out)
        self.assertEqual(output.shape, torch.Size([3, 7, 9]))  # batch, tgt_len, vocab_size

    # Check that if occlusion_masks are provided, they are used instead of generating new ones
    def test_forward_with_without_masks(self):
        provided_masks = torch.randn(3, 7, 512)

        # masks are not provided
        output, model_specific_output = self.decoder.forward(
            self.prev_output_tokens, self.encoder_out)
        self.assertFalse(torch.allclose(model_specific_output["masks"], provided_masks))

        # masks are provided
        output, model_specific_output = self.decoder.forward(
            self.prev_output_tokens, self.encoder_out, occlusion_masks=provided_masks)
        self.assertTrue(torch.allclose(model_specific_output["masks"], provided_masks))

    # Check that if last_tokens_to_perturb is provided, no occlusion happens after those positions
    def test_forward_last_tokens_to_perturb(self):
        output, model_specific_output = self.decoder.forward(
            self.prev_output_tokens, self.encoder_out, last_tokens_to_perturb=[3, 3, 3])
        self.assertTrue(torch.all(model_specific_output["masks"][:, 4:]))


if __name__ == '__main__':
    unittest.main()
