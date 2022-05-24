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
import argparse
import ast
import csv
import logging
import os
import os.path as op

import numpy as np
import torch
from ctc_segmentation import CtcSegmentationParameters, ctc_segmentation, \
    prepare_token_list
from torch.nn import functional as F

from examples.speech_to_text.data.speech_to_text_dataset_with_src import S2TDataConfigSrc
from fairseq import utils, options, tasks, checkpoint_utils
from fairseq.data import Dictionary, encoders
from fairseq.data.audio.speech_to_text_dataset import SpeechToTextDatasetCreator, SpeechToTextDataset
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from fairseq.models import FairseqEncoderDecoderModel

CTC_BLANK = "<ctc_blank>"


def read_data(cfg, task, logger):
    """
    Reads the data from the transcripts file and the tsv definition and
    returns batches whose target is the provided transcript and the source
    is the audio.
    """
    tsv_path = op.join(cfg.task.data, f"{cfg.dataset.gen_subset}.tsv")
    data_cfg = S2TDataConfigSrc(op.join(cfg.task.data, cfg.task.config_yaml))
    if not op.isfile(tsv_path):
        raise FileNotFoundError(f"Dataset not found: {tsv_path}")
    with open(tsv_path) as f:
        reader = csv.DictReader(
            f,
            delimiter="\t",
            quotechar=None,
            doublequote=False,
            lineterminator="\n",
            quoting=csv.QUOTE_NONE,
        )
        audio_paths = []
        n_frames = []
        for e in reader:
            audio_paths.append(e[SpeechToTextDatasetCreator.KEY_AUDIO])
            n_frames.append(int(e[SpeechToTextDatasetCreator.KEY_N_FRAMES]))

    with open(cfg.task.text_file) as f:
        forced_strings = f.readlines()

    # Source dict
    source_dict_path = op.join(cfg.task.data, data_cfg.vocab_filename_src)
    src_dict = Dictionary.load(source_dict_path)
    src_dict.add_symbol(CTC_BLANK)
    logger.info(
        f"source dictionary size ({data_cfg.vocab_filename_src}): " f"{len(src_dict):,}"
    )
    dataset = SpeechToTextDataset(
        cfg.dataset.gen_subset,
        False,
        data_cfg,
        audio_paths,
        n_frames,
        tgt_dict=src_dict,
        tgt_texts=forced_strings,
        bpe_tokenizer=encoders.build_bpe(argparse.Namespace(**data_cfg.bpe_tokenizer_src)))

    # return a batch iterator
    itr = task.get_batch_iterator(
        dataset=dataset,
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_shards=cfg.distributed_training.distributed_world_size,
        shard_id=cfg.distributed_training.distributed_rank,
        num_workers=cfg.dataset.num_workers,
        data_buffer_size=cfg.dataset.data_buffer_size,
    ).next_epoch_itr(shuffle=False)
    return dataset, progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
    )


def split_token_list(ltokens, split_chars=None):
    """
    Given a list of tokens and a set of splitting ids, it returns a list of list of tokens,
    by splitting the original list when a splitting id is met.

    >>> split_token_list([1, 2, 3, 4, 5], split_chars={3})
    [[1, 2, 3], [4, 5]]
    >>> split_token_list([1, 2, 3, 4, 5], split_chars={2, 3})
    [[1, 2], [3], [4, 5]]
    >>> split_token_list([1, 2, 3, 4, 5], split_chars={2, 3, 5})
    [[1, 2], [3], [4, 5]]
    >>> split_token_list([1, 2, 3, 4, 5])
    [[1, 2, 3, 4, 5]]
    >>> split_token_list([1, 2, 3, 4, 5], split_chars={30})
    [[1, 2, 3, 4, 5]]
    >>> split_token_list([], split_chars={30})
    [[]]
    """
    splits = [[]]
    if split_chars is None:
        split_chars = {}
    for tok in ltokens:
        splits[-1].append(tok)
        if tok in split_chars:
            splits.append([])
    if len(splits[-1]) == 0 and len(splits) > 1:
        splits.pop()
    return splits


def determine_time(utt_begin_indices, timings, text):
    """
    Determine timestamps without making the average as the built-in function.
    :param utt_begin_indices: list of time indices of utterance start
    :param timings: mapping of time indices to seconds
    :param text: list of utterances
    :return: segments, a list of utterance start and end [s]
    """
    segments = []
    for i in range(len(text)):
        start = timings[utt_begin_indices[i]]
        # If no split_chars has been produced, return the start time
        if len(utt_begin_indices) == 1:
            end = start
        else:
            end = timings[utt_begin_indices[i + 1]]
        segments.append((start, end))
    return segments


def main(cfg):
    if isinstance(cfg, argparse.Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    assert cfg.common_eval.path is not None, "--path required for generation!"
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
    )
    logger = logging.getLogger(__name__)
    utils.import_user_module(cfg.common)

    if cfg.dataset.max_tokens is None and cfg.dataset.batch_size is None:
        cfg.dataset.max_tokens = 12000
    logger.info(cfg)
    # Load dataset splits
    task = tasks.setup_task(cfg.task)
    overrides = ast.literal_eval(cfg.common_eval.model_overrides)
    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    # Load ensemble
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    models, saved_cfg = checkpoint_utils.load_model_ensemble(
        utils.split_paths(cfg.common_eval.path),
        arg_overrides=overrides,
        task=task,
        suffix=cfg.checkpoint.checkpoint_suffix,
        strict=(cfg.checkpoint.checkpoint_shard_count == 1),
        num_shards=cfg.checkpoint.checkpoint_shard_count,
    )
    assert len(models) == 1, "only 1 model is currently supported"
    model = models[0]
    assert isinstance(model, FairseqEncoderDecoderModel)
    assert model.encoder.ctc_flag, "encoder should be CTC-aware"
    if cfg.common.fp16:
        model.half()
    if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
        model.cuda()
    model.prepare_for_inference_(cfg)
    encoder = model.encoder

    ds, itr = read_data(cfg, task, logger)

    split_chars = None
    if cfg.task.split_tokens is not None:
        split_chars = {ds.tgt_dict.index(token) for token in cfg.task.split_tokens.split(",")}

    ctc_config = CtcSegmentationParameters(
        char_list=ds.tgt_dict.symbols,
        blank=ds.tgt_dict.index(CTC_BLANK),
        index_duration=cfg.task.feature_duration,
        start_of_ground_truth="Ꚃ",
    )
    ctc_config.update_excluded_characters()

    for sample in itr:
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        encoder_out = encoder(**sample["net_input"])
        ctc_out = encoder_out["ctc_out"]
        ctc_lengths = encoder_out["ctc_lengths"]
        lprob = F.log_softmax(ctc_out, dim=-1).transpose(0, 1)
        for i, sample_id in enumerate(sample["id"].tolist()):
            tgt_len = sample["target_lengths"][i].item()
            tgt_tokens = sample["target"][i][:tgt_len - 1].tolist()
            if tgt_len <= 1:
                # if string to match is empty (ie. contains only EOS), return start and end time of the segment
                timings = []
                segments = [(0.0, ctc_lengths[i].item() * cfg.task.feature_duration)]
            else:
                token_splits = split_token_list(tgt_tokens, split_chars=split_chars)
                tokenized_texts = [np.array(l) for l in token_splits]
                ground_truth_mat, utt_begin_indices = prepare_token_list(ctc_config, tokenized_texts)
                logging.debug(f"{tokenized_texts} have been converted into the ground matrix {ground_truth_mat}")
                # Align using CTC segmentation
                lpz = lprob[i][:ctc_lengths[i].item()].cpu().detach().numpy()
                if len(ground_truth_mat) > lpz.shape[0]:
                    timings = []
                    segments = [(0.0, ctc_lengths[i].item() * cfg.task.feature_duration)]
                    logging.warning("Attention: the text is longer than audio and cannot be aligned.")
                else:
                    timings, char_probs, state_list = ctc_segmentation(ctc_config, lpz, ground_truth_mat)
                    # Obtain list of utterances with time intervals
                    segments = determine_time(utt_begin_indices, timings, token_splits)
            print(f"TEXT-{sample_id}\t{ds.tgt_dict.string(tgt_tokens)}")
            print(f"TIME-{sample_id}\t{' '.join(np.char.mod('%f', timings))}")
            print(f"SEGM-{sample_id}\t{' '.join(str(start) + '-' + str(end) for start, end in segments)}")


if __name__ == '__main__':
    parser = options.get_generation_parser()
    parser.add_argument('--text-file', type=str, required=True,
                        help="Texts to be used for the alignment (one per line)")
    parser.add_argument('--split-tokens', type=str, required=False, default=None,
                        help="tokens used to split the utterances")
    parser.add_argument('--feature-duration', type=float, default=0.040,
                        help="the time (in seconds) a single output of the CTC corresponds to")
    main(options.parse_args_and_arch(parser))
