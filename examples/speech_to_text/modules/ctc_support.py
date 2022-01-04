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
import math
from itertools import groupby

import torch
from torch import nn
from torch.nn import functional as F

import logging

from fairseq.data.data_utils import lengths_to_padding_mask

logger = logging.getLogger(__name__)


class CtcSupport:
    FIXED_RATIO = 4
    """
    This class adds the CTC loss computation (enabled by "ctc_multi_loss" criterion) to the model by adding a Linear
    layer on the layer specified by --ctc-encoder-layer. By specifying the CTC compression strategy via
    --ctc-compress-strategy, the CTC compression is applied to the input.
    """

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--ctc-compress-strategy",
            type=str,
            default="none",
            choices=['none', 'avg', 'weighted', 'softmax', 'fixed'],
            help="Strategy to use when compressing CTC output"
        )
        parser.add_argument(
            "--ctc-compress-fixed-ratio",
            type=int,
            default=4,
            help="if --ctc-compress-strategy is set to fixed, this "
                 "parameter controls how many consecutive steps to merge"
        )
        parser.add_argument(
            '--ctc-compress-max-out-size', type=int, default=-1,
            help="if CTC compression is enabled and this argument is set to a positive number, "
                 "every input is forced to be at most as long as the value set for this parameter, "
                 "even though the CTC would not compress it enough. Intuitively, this parameter "
                 "should be set to 1/4 of the max input length to ensure that the maximum sequence "
                 "length of the Transformer self-attention input is the same as in the case of "
                 "of models having 2 initial convolutions with stride 2."
        )

    def ctc_init(self, args, src_dictionary):
        self.ctc_flag = False
        if args.criterion == "ctc_multi_loss" or args.ctc_compress_strategy != "none":
            self.ctc_flag = True
        if self.ctc_flag:
            self.ctc_fc = nn.Linear(args.encoder_embed_dim, len(src_dictionary))
            self.ctc_layer = args.ctc_encoder_layer
            if args.ctc_compress_strategy != "none":
                self.ctc_compress_method = getattr(CTCCompressStrategy, args.ctc_compress_strategy)
                self.ctc_compress_max_out_size = args.ctc_compress_max_out_size
                CtcSupport.FIXED_RATIO = args.ctc_compress_fixed_ratio
            else:
                self.ctc_compress_method = "none"

    def apply_ctc(self, x, input_lengths):
        x_ctc = self.ctc_fc(x)
        if self.ctc_compress_method != "none":
            x, input_lengths = self.average_same_ctc_features(x_ctc, x, input_lengths)
        encoder_padding_mask = lengths_to_padding_mask(input_lengths)
        return x, x_ctc, encoder_padding_mask

    def ctc_encoder_out(self, encoder_out_dict, x_ctc, ctc_lengths):
        if self.ctc_flag:
            encoder_out_dict["ctc_out"] = x_ctc  # T x B x D
            encoder_out_dict["ctc_lengths"] = ctc_lengths
        return encoder_out_dict

    def average_same_ctc_features(self, x_ctc, x, src_lengths):
        with torch.no_grad():
            batch_predicted = []
            prob_ctc = F.softmax(x_ctc, dim=-1).transpose(0, 1)  # from T x B x D to B x T x D
            for b in range(prob_ctc.shape[0]):
                predicted = prob_ctc[b][: src_lengths[b]].argmax(-1).tolist()
                batch_predicted.append([(p[0], len(list(p[1]))) for p in groupby(predicted)])
            batch_predicted = self.ensure_max_ctc_out_len(batch_predicted)
            weights_matrix, new_lengths = self.ctc_compress_method(
                prob_ctc, batch_predicted, x.dtype, x.device)
        # x is T x B x C -> B x C x T; weights_matrix is B x T x T'
        compressed_output = x.permute(1, 2, 0).bmm(weights_matrix)  # B x C x T'
        return compressed_output.permute(2, 0, 1), src_lengths.new(new_lengths)

    def ensure_max_ctc_out_len(self, batch_predicted):
        """
        Ensures that the output of the CTC compression is not longer than the ctc_compress_max_out_size.
        If there are samples violating this constraints, consecutive predictions are merged
        so to shorten the sentence.
        E.g. if the ctc_compress_max_out_size is set to 3, and the output of the CTC compression would be
        long 5, the first and second predictions are merged, as well as the third and the fourth. So, the
        corresponding vectors will be merged according to the CTC compression strategy.
        """
        if self.ctc_compress_max_out_size > 0:

            def merge_sublist(elements):
                """
                Takes a list of Tuples (predicted_element, num_corresponding_vectors) and returns
                a single tuple with the predicted_element having the highest number of corresponding_vectors
                (in case of a tie, the first is returned) and the total sum of the num_corresponding_vectors
                E.g. if the input is [(a, 3), (b, 5), (c, 6), (a, 4)], the output will be (a, 18).
                """
                sum_num_vectors = 0
                max_element = None
                max_element_cnt = 0
                temp_dict = {}
                for predicted_element, num_corresponding_vectors in elements:
                    if predicted_element in temp_dict:
                        temp_dict[predicted_element] += num_corresponding_vectors
                    else:
                        temp_dict[predicted_element] = num_corresponding_vectors
                    if temp_dict[predicted_element] > max_element_cnt:
                        max_element_cnt = temp_dict[predicted_element]
                        max_element = predicted_element
                    sum_num_vectors += num_corresponding_vectors
                return max_element, sum_num_vectors

            for b_idx, p in enumerate(batch_predicted):
                pred_len = len(p)
                if pred_len > self.ctc_compress_max_out_size:
                    reduction_factor = math.ceil(pred_len / self.ctc_compress_max_out_size)
                    i = 0
                    new_p = []
                    while i < pred_len:
                        new_p.append(merge_sublist(p[i:i + reduction_factor]))
                        i += reduction_factor
                    batch_predicted[b_idx] = new_p

        return batch_predicted

    def reorder_ctc(self, reordered_dict, encoder_out, new_order):
        if self.ctc_flag:
            new_ctc_out = encoder_out["ctc_out"].index_select(1, new_order)
            new_ctc_lengths = encoder_out["ctc_lengths"].index_select(0, new_order)
            reordered_dict["ctc_out"] = new_ctc_out  # T x B x D
            reordered_dict["ctc_lengths"] = new_ctc_lengths
        return reordered_dict


class CTCCompressStrategy:
    @staticmethod
    def new_lengths(batch_predicted):
        return [len(p) for p in batch_predicted]

    @staticmethod
    def avg(prob_ctc, predicted, dtype, device):
        new_lengths = CTCCompressStrategy.new_lengths(predicted)
        new_maxlen = max(new_lengths)
        weights_matrix = torch.zeros((prob_ctc.shape[0], prob_ctc.shape[1], new_maxlen), dtype=dtype)
        for b_idx, pred in enumerate(predicted):
            processed_inputs_cnt = 0
            for t_idx, same in enumerate(pred):
                new_processed_inputs_cnt = processed_inputs_cnt + same[1]
                weights_matrix[b_idx, processed_inputs_cnt:new_processed_inputs_cnt, t_idx] = 1.0 / same[1]
                processed_inputs_cnt = new_processed_inputs_cnt
        return weights_matrix.to(device), new_lengths

    @staticmethod
    def weighted(prob_ctc, predicted, dtype, device):
        new_lengths = CTCCompressStrategy.new_lengths(predicted)
        new_maxlen = max(new_lengths)
        weights_matrix = torch.zeros((prob_ctc.shape[0], prob_ctc.shape[1], new_maxlen), dtype=dtype, device=device)
        for b_idx, pred in enumerate(predicted):
            processed_inputs_cnt = 0
            for t_idx, same in enumerate(pred):
                new_processed_inputs_cnt = processed_inputs_cnt + same[1]
                # Get the probabilities of the prediction for the different time steps as weight
                weights = prob_ctc[b_idx, processed_inputs_cnt:new_processed_inputs_cnt, same[0]]
                weights_matrix[b_idx, processed_inputs_cnt:new_processed_inputs_cnt, t_idx] = \
                    weights / weights.sum()
                processed_inputs_cnt = new_processed_inputs_cnt
        return weights_matrix, new_lengths

    @staticmethod
    def softmax(prob_ctc, predicted, dtype, device):
        new_lengths = CTCCompressStrategy.new_lengths(predicted)
        new_maxlen = max(new_lengths)
        weights_matrix = torch.zeros((prob_ctc.shape[0], prob_ctc.shape[1], new_maxlen), dtype=dtype, device=device)
        for b_idx, pred in enumerate(predicted):
            processed_inputs_cnt = 0
            for t_idx, same in enumerate(pred):
                new_processed_inputs_cnt = processed_inputs_cnt + same[1]
                # Get the probabilities of the prediction for the different time steps as weight
                weights = F.softmax(prob_ctc[b_idx, processed_inputs_cnt:new_processed_inputs_cnt, same[0]])
                weights_matrix[b_idx, processed_inputs_cnt:new_processed_inputs_cnt, t_idx] = \
                    weights / weights.sum()
                processed_inputs_cnt = new_processed_inputs_cnt
        return weights_matrix, new_lengths

    @staticmethod
    def fixed(prob_ctc, predicted, dtype, device):
        new_maxlen = math.ceil(prob_ctc.shape[1] / CtcSupport.FIXED_RATIO)
        weights_matrix = torch.zeros((prob_ctc.shape[0], prob_ctc.shape[1], new_maxlen), dtype=dtype)
        new_lengths = []
        for b_idx, pred in enumerate(predicted):
            original_len = sum(x[1] for x in pred)
            new_len = 0
            for new_t_idx in range(new_maxlen):
                processed_inputs_cnt = new_t_idx * CtcSupport.FIXED_RATIO
                processed_inputs_cnt_end = processed_inputs_cnt + CtcSupport.FIXED_RATIO
                if processed_inputs_cnt_end > original_len:
                    processed_inputs_cnt_end = original_len
                weights_matrix[b_idx, processed_inputs_cnt:processed_inputs_cnt_end, new_t_idx] = \
                    1.0 / (processed_inputs_cnt_end - processed_inputs_cnt)
                new_len += 1
                if processed_inputs_cnt_end == original_len:
                    break
            new_lengths.append(new_len)
        return weights_matrix.to(device), new_lengths
