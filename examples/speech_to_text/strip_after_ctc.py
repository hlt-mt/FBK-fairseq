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

from collections import OrderedDict
import argparse

from fairseq import utils
from fairseq.checkpoint_utils import torch_persistent_save, load_checkpoint_to_cpu


def _strip_params(state, strip_what):
    new_state = state
    new_state['model'] = OrderedDict(
        {key: value for key, value in state['model'].items() if not key.startswith(strip_what)})

    return new_state


def save_state(state, filename):
    torch_persistent_save(state, filename)


def main(args):
    utils.import_user_module(args)
    model_state = load_checkpoint_to_cpu(args.model_path)
    print("Loaded model {}".format(args.model_path))
    strip_what = [f'encoder.linformer_layers.{num}' for num in range(args.ctc_encoder_layer,args.num_encoder_layers)]
    #strip_what.extend(['ctc_fc'])
    model_state = _strip_params(model_state, strip_what=tuple(strip_what))
    print("Stripped {}".format(strip_what))
    save_state(model_state, args.new_model_path)
    print("Saved to {}".format(args.new_model_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--user-dir', default=None,
                        help='path to a python module containing custom extensions (tasks and/or architectures)')
    parser.add_argument('--model-path', type=str, required=True,
                        help="The path to the model to strip")
    parser.add_argument('--new-model-path', type=str, required=True,
                        help="The name for the stripped model")
    parser.add_argument('--ctc-encoder-layer', type=int, default=8,
                        help="Number of layer to which ctc is applied")
    parser.add_argument('--num-encoder-layers', type=int, default=12,
                        help="Number of encoder layers")
    main(parser.parse_args())
