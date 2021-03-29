import csv
import os.path as op
import sys
from argparse import Namespace

from examples.speech_to_text.data.speech_to_text_dataset_with_src import S2TDataConfigSrc

from examples.speech_to_text.data.speech_to_text_dataset_KD import SpeechToTextDatasetCreatorKD
from fairseq.data import encoders, Dictionary

root = sys.argv[1]
split = sys.argv[2]
config_yaml = sys.argv[3]

tsv_path = op.join(root, f"{split}.tsv")
teacher_path = op.join(root, f"prob_idx_{split}.tsv")
is_train_split = split.startswith("train")
data_cfg = S2TDataConfigSrc(op.join(root, config_yaml))
pre_tokenizer = encoders.build_tokenizer(Namespace(**data_cfg.pre_tokenizer))
bpe_tokenizer = encoders.build_bpe(Namespace(**data_cfg.bpe_tokenizer))
bpe_tokenizer_src = encoders.build_bpe(Namespace(**data_cfg.bpe_tokenizer_src))
src_dict = Dictionary.load(op.join(root, data_cfg.vocab_filename_src))
tgt_dict = Dictionary.load(op.join(root, data_cfg.vocab_filename))

with open(tsv_path) as f:
    reader = csv.DictReader(
        f,
        delimiter="\t",
        quotechar=None,
        doublequote=False,
        lineterminator="\n",
        quoting=csv.QUOTE_NONE,
    )
    samples = [dict(e) for e in reader]
    assert len(samples) > 0
with open(teacher_path) as f:
    reader = csv.DictReader(
        f,
        delimiter="\t",
        quotechar=None,
        doublequote=False,
        lineterminator="\n",
        quoting=csv.QUOTE_NONE,
    )
    prob_idx = [dict(e) for e in reader]
    assert len(prob_idx) > 0


dataset = SpeechToTextDatasetCreatorKD._from_list(
        split,
        is_train_split,
        [samples],
        [prob_idx],
        data_cfg,
        tgt_dict,
        src_dict,
        pre_tokenizer,
        bpe_tokenizer,
        bpe_tokenizer_src,
    )


with open(tsv_path) as f, open(tsv_path.replace(".tsv", "_filtered.tsv"), "w") as fw, \
        open(teacher_path) as t, open(teacher_path.replace(".tsv", "_filtered.tsv"), "w") as tw:
    lines = f.readlines()
    fw.write(lines[0])
    teach_lines = t.readlines()
    tw.write(teach_lines[0])
    for i in range(len(dataset)):
        s = dataset[i]
        if s[2].shape[0] - 1 == s[4].shape[0]:
            fw.write(lines[i+1])
            tw.write(teach_lines[i+1])
