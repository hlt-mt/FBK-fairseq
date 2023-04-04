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
from fairseq import checkpoint_utils
import logging

from torch.nn import Module
from torch.nn.modules.batchnorm import _BatchNorm

logger = logging.getLogger(__name__)


class EncoderPretrainingSupport(Module):
    """
    This class implements the loading of a pre-trained encoder or a part of it:
    - if --load-pretrained-encoder-from is specified, the whole encoder will be loaded from the path and an error arises
    if the encoder has a different structure than the current architecture
    - if --allow-partial-encoder-loading is enabled, only a part of encoder can be loaded (e.g. 2 out of 4 layers)
    without issuing any error
    - if --freeze-encoder is enabled, the encoder weights are not updated during training
    """

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--load-pretrained-encoder-from",
            type=str,
            metavar="STR",
            help="model to take encoder weights from (for initialization)",
        )
        parser.add_argument(
            '--allow-partial-encoder-loading',
            action='store_true',
            default=False,
            help="if set, the model is restored even if it doesn't match exactly the architecture, "
                 "ie. some params are missing."
        )
        parser.add_argument(
            '--freeze-encoder',
            action='store_true',
            default=False,
            help="If set, the encoder is not trained together with the model and the weights"
                 "remains freezed."
        )

    @staticmethod
    def load_pretrained(args, encoder):
        if getattr(args, "load_pretrained_encoder_from", None):
            encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder, checkpoint=args.load_pretrained_encoder_from,
                allow_partial_encoder_loading=getattr(args, "allow_partial_encoder_loading", False),
            )
            logger.info(
                f"loaded pretrained encoder from: "
                f"{args.load_pretrained_encoder_from}"
            )
        if getattr(args, "freeze_encoder", False):
            for _, param in encoder.named_parameters():
                param.requires_grad = False
            # set BatchNorm layers in eval mode to avoid changes
            for module in encoder.modules():
                if isinstance(module, _BatchNorm):
                    module.eval()
            setattr(encoder, '__freeze_batchnorm', True)
        return encoder

    def train(self, mode: bool = True):
        super().train(mode)
        # set BatchNorm layers in eval mode to avoid changes when they have
        # to be freezed. This is required as train() is called on the model
        # before each epoch starts and after begin_epoch() is called on the
        # task.
        if getattr(self.encoder, "__freeze_batchnorm", False) and mode:
            for module in self.encoder.modules():
                if isinstance(module, _BatchNorm):
                    module.eval()
        return self
