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

logger = logging.getLogger(__name__)


class EncoderPretrainingSupport:
    """
    This class implements the loading of a pre-trained encoder or a part of it:
    - if --load-pretrained-encoder-from is specified, the whole encoder will be loaded from the path and an error arises
    if the encoder has a different structure than the current architecture
    - if --allow-partial-encoder-loading is enabled, only a part of encoder can be loaded (e.g. 2 out of 4 layers)
    without issuing any error
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
        return encoder
