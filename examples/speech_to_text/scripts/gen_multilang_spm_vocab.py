# Copyright 2022 FBK
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

from pathlib import Path
from tempfile import NamedTemporaryFile

import argparse
from examples.speech_to_text.data_utils_new import load_df_from_tsv, gen_vocab


def gen_multilang_spm_vocab(args):
    langs = args.langs.strip().split(",")
    splits = args.splits.strip().split(",")
    with NamedTemporaryFile(mode="w") as f:
        for split in splits:
            tsv_path = args.data_root + f"{split}.tsv"
            df = load_df_from_tsv(tsv_path)
            for t in df["tgt_text"]:
                f.write(t + "\n")
        special_symbols = [f'<lang:{lang}>' for lang in langs]
        spm_prefix = f"spm_{args.vocab_type}{args.vocab_size}_multi"
        gen_vocab(
            Path(f.name),
            Path(args.save_dir + spm_prefix),
            args.vocab_type,
            args.vocab_size,
            special_symbols=special_symbols
        )


def main():
    """
    The script generates a multilingual SentencePiece vocabulary with its SentencePiece model
    starting from tsv files contained in **data_root** and defined through **splits**.
    The language tags, useful for multilingual training, are added as special symbols to
    the SentencePiece vocabulary and are defined by the user through **langs** (separated by commas).
    **vocab_type** defines the vocabulary type of the SentencePiece model and
    **vocab_size** defines its dimension.
    **save_dir** will contain the SentencePiece model (.model), the SentencePiece
    vocabulary (.vocab) and its correspondent Fairseq dictionary (.txt) that is
    necessary for model training.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-data", required=True, type=str)
    parser.add_argument("--save-dir", "-save", required=True, type=str)
    parser.add_argument("--langs", "-l", required=True, type=str)
    parser.add_argument("--splits", "-splits", required=True, type=str)
    parser.add_argument("--vocab-type", default="unigram", type=str,
                        choices=["bpe", "unigram", "char"])
    parser.add_argument("--vocab-size", default=10000, type=int)
    args = parser.parse_args()

    gen_multilang_spm_vocab(args)


if __name__ == "__main__":
    main()
