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
import logging
from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F

try:
    from torch.masked import mean as masked_mean
except ImportError:
    def masked_mean(x: Tensor, mask: Optional[Tensor] = None, dim=0):
        assert mask is not None
        masked_sum = (x * mask).sum(dim=dim)
        masked_total = mask.sum(dim=dim)
        return masked_sum / masked_total


from examples.speech_to_text.modules.gradient_reversal import GradientReversalLayer
from fairseq import checkpoint_utils
from fairseq.models import FairseqEncoderDecoderModel


logger = logging.getLogger(__name__)


class MultiTaskModel(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder, auxiliary_decoder):
        super().__init__(encoder, decoder)
        self.auxiliary_decoder = auxiliary_decoder

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out)
        auxiliary_out = self.auxiliary_decoder(encoder_out)
        if self.encoder.ctc_flag:
            return (decoder_out, auxiliary_out), {
                "ctc_out": encoder_out["ctc_out"], "ctc_lengths": encoder_out["ctc_lengths"]}
        else:
            return decoder_out, auxiliary_out

    def get_auxiliary_target(self, sample, auxiliary_output):
        return sample["auxiliary_target"]


class MultiTaskClassifierModel(MultiTaskModel):

    def __init__(self, encoder, decoder, auxiliary_decoder):
        super().__init__(encoder, decoder, auxiliary_decoder)
        self.__freeze_base = False
        self.__freeze_classifier = False

    @staticmethod
    def add_args(parser):
        parser.add_argument("--reverted-classifier", action='store_true', default=False,
                            help="if set, the gradient of the classifier is inverted")
        parser.add_argument("--reverted-lambda", type=float, required=False,
                            help="if set, the gradient reversal factor is fixed and set to this value")
        parser.add_argument("--reverted-gamma", type=float, required=False, default=10,
                            help="if --reverted-classifier is used, and --reverted-lambda is not set "
                                 "it controls how fast the gradient scaling factor increase")
        parser.add_argument('--pretrained-model', type=str, default=None,
                            help='path to a pretrained ST model')
        parser.add_argument('--freeze-model', action='store_true', default=False,
                            help='if set, the base model is freezed')

    def freeze_base_model(self, update_weights=False):
        """
        Freezes the base model weights. Is update_weights is set to True,
        the effect is, instead, the opposite.
        """
        for _, param in self.encoder.named_parameters():
            param.requires_grad = update_weights
        for _, param in self.decoder.named_parameters():
            param.requires_grad = update_weights
        # set BatchNorm layers in eval mode to avoid changes
        for module in [*self.encoder.modules(), *self.decoder.modules()]:
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                if update_weights:
                    module.train()
                else:
                    module.eval()
        self.__freeze_base = not update_weights

    def freeze_classifier(self, update_weights=False):
        """
        Freezes the classifier weights. Is update_weights is set to True,
        the effect is, instead, the opposite.
        """
        for _, param in self.auxiliary_decoder.named_parameters():
            param.requires_grad = update_weights
        self.__freeze_classifier = not update_weights

    def train(self, mode: bool = True):
        super().train(mode)
        # set BatchNorm layers in eval mode to avoid changes when they have
        # to be freezed. This is required as train() is called on the model
        # before each epoch starts and after begin_epoch() is called on the
        # task.
        if self.__freeze_base and mode:
            for module in [*self.encoder.modules(), *self.decoder.modules()]:
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.eval()
        if self.__freeze_classifier and mode:
            for module in self.auxiliary_decoder.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    @classmethod
    def build_with_classifier(cls, base_model, args, task):
        num_outputs = len(task.data_cfg.aux_classes)
        if args.reverted_classifier:
            auxiliary_decoder = RevertedFFNDecoderClassifier(args, num_outputs)
        else:
            auxiliary_decoder = FFNDecoderClassifier(args, num_outputs)
        encoder = base_model.encoder
        decoder = base_model.decoder
        if getattr(args, "pretrained_model", None):
            encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder, checkpoint=args.pretrained_model,
                allow_partial_encoder_loading=False, )
            decoder = checkpoint_utils.load_pretrained_component_from_model(
                component=decoder, checkpoint=args.pretrained_model,
                allow_partial_encoder_loading=False, )
            logger.info(f"loaded pretrained model from: {args.pretrained_model}")
        model = cls(encoder, decoder, auxiliary_decoder)
        if getattr(args, "freeze_model", False):
            model.freeze_base_model()
        return model


class ClassifierDecoder(nn.Module):
    def get_normalized_probs(self, net_output, log_probs=False):
        if self.output_size == 1:
            if log_probs:
                return F.logsigmoid(net_output)
            else:
                return F.sigmoid(net_output)
        else:
            if log_probs:
                return F.log_softmax(net_output, dim=-1)
            else:
                return F.softmax(net_output, dim=-1)


class FFNDecoderClassifier(ClassifierDecoder):
    def __init__(self, args, output_size):
        super().__init__()
        self.output_size = output_size
        self.input_size = args.encoder_embed_dim
        self.fc1 = nn.Linear(self.input_size, self.input_size * 2)
        self.fc2 = nn.Linear(self.input_size * 2, self.output_size)

    def do_forward(self, x, padding_mask):
        x = self.fc2(F.relu(self.fc1(x)))  # T x B x C
        if padding_mask is not None:
            return masked_mean(
                x, mask=padding_mask.logical_not().transpose(0, 1).unsqueeze(-1), dim=0)
        else:
            return torch.mean(x, dim=0)

    def forward(self, encoder_out):
        x = encoder_out["encoder_out"][0]  # T x B x C
        if len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"][0]
        else:
            padding_mask = None
        return self.do_forward(x, padding_mask)


class RevertedFFNDecoderClassifier(FFNDecoderClassifier):
    def __init__(self, args, output_size):
        super(RevertedFFNDecoderClassifier, self).__init__(args, output_size)
        self.max_updates = getattr(args, 'max_update', 100000)
        self.reverted_lambda = getattr(args, 'reverted_lambda', None)
        if self.reverted_lambda is not None:
            self.gradient_reversal = GradientReversalLayer(lambda_factor=self.reverted_lambda)
        else:
            self.gradient_reversal = GradientReversalLayer(
                max_updates=self.max_updates,
                gamma=getattr(args, 'reverted_gamma', 10),
            )

    def do_forward(self, x, padding_mask):
        return super().do_forward(self.gradient_reversal(x), padding_mask)
