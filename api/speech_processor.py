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
import ast
import logging
import time

import dataclasses
import torch

from api.ne_postprocessing import move_tags_after_space, move_tags_to_start_or_end
from examples.speech_to_text.data_utils_new import extract_fbank_features
from examples.speech_to_text.utils.tags import join_tags_tokens
from fairseq import checkpoint_utils, tasks, utils
from fairseq.data.audio.audio_utils import get_waveform
from fairseq.data.audio.feature_transforms import CompositeAudioFeatureTransform


class SpeechToTextProcessor:
    """
    Base class to process requests with audio input and textual output.
    """

    # The dataclass representing the request
    request_class: type

    def __init__(self, cfg):
        self.logger = logging.getLogger(self.__class__.__name__)
        utils.import_user_module(cfg.common)
        self.logger.info(cfg)
        self.use_cuda = torch.cuda.is_available() and not cfg.common.cpu

        self.task = tasks.setup_task(cfg.task)

        # Set dictionaries
        self.src_dict = getattr(self.task, 'source_dictionary', None)
        self.tgt_dict = self.task.target_dictionary

        self.models = self.load_models(cfg)

        self.generator = self.task.build_generator(
            self.models, cfg.generation
        )
        self.feature_transforms = CompositeAudioFeatureTransform.from_config_dict(
            self.task.data_cfg.get_feature_transforms("test", False)
        )
        self.nbest = cfg.generation.nbest
        # Handle tokenization and BPE
        self.tokenizer = self.task.build_tokenizer(cfg.tokenizer)
        self.bpe = self.task.build_bpe(cfg.bpe)
        self.post_process = cfg.common_eval.post_process

    def decode_fn(self, x):
        if self.bpe is not None:
            x = self.bpe.decode(x)
        if self.tokenizer is not None:
            x = self.tokenizer.decode(x)
        return x

    def load_models(self, cfg):
        overrides = ast.literal_eval(cfg.common_eval.model_overrides)

        # Load ensemble
        self.logger.info("loading model(s) from {}".format(cfg.common_eval.path))
        models, saved_cfg = checkpoint_utils.load_model_ensemble(
            utils.split_paths(cfg.common_eval.path),
            arg_overrides=overrides,
            task=self.task,
            suffix=cfg.checkpoint.checkpoint_suffix,
            strict=(cfg.checkpoint.checkpoint_shard_count == 1),
            num_shards=cfg.checkpoint.checkpoint_shard_count,
        )

        # Optimize ensemble for generation
        for model in models:
            if model is None:
                continue
            if cfg.common.fp16:
                model.half()
            if self.use_cuda and not cfg.distributed_training.pipeline_model_parallel:
                model.cuda()
            model.prepare_for_inference_(cfg)
        return models

    def preproc_audio(self, audio_fn):
        start_time = time.time()
        waveform, sample_rate = get_waveform(audio_fn)
        source = extract_fbank_features(torch.from_numpy(waveform), sample_rate)
        if self.feature_transforms is not None:
            source = self.feature_transforms(source)
        source = torch.from_numpy(source).float()
        end_time = time.time()
        self.logger.info(f"Preprocessing of {audio_fn} took {end_time - start_time} s.")
        return source

    def process(self, request_id, request):
        """
        Elaborates the provided request.
        """
        self.logger.debug(f"Received request: ID[{request_id}] - {request}")
        parsed_request = self.request_class(**request)
        start_time = time.time()
        input_audio = self.preproc_audio(parsed_request.wav_path)
        sample = {
            'net_input': {
                'src_tokens': input_audio.unsqueeze(0),
                'src_lengths': torch.Tensor([input_audio.shape[0]])
            }
        }
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Request ID[{request_id}] net input: {sample}")
        inference_start_time = time.time()
        sample = utils.move_to_cuda(sample) if self.use_cuda else sample
        hypos = self.task.inference_step(self.generator, self.models, sample)
        inference_end_time = time.time()
        assert len(hypos) == 1, "generation of multiple sentences not supported"
        hypo = hypos[0][0]  # We consider only the most likely hypothesis
        self.logger.info(
            f"Inference for Request ID[{request_id}] took: {inference_end_time - inference_start_time} s.")

        result = self._postproc(request_id, hypo, parsed_request)
        result = dataclasses.asdict(result)
        end_time = time.time()
        self.logger.info(f"Request ID[{request_id}] processed in {end_time - start_time} s.")
        return result

    def _postproc(self, request_id, hypo, request):
        """
        Abstract method to be implemented by subclasses.
        This method should return a dataclass.
        """
        pass

    def _postproc_out_and_tags(self, request_id, tokens, dictionary, tags, symbols_to_ignore, ttype="target"):
        if tags is not None:
            tags_strings, joint_string = join_tags_tokens(
                tags.int().cpu(), tokens, dictionary, self.task.data_cfg.tags)

            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"Request ID[{request_id}] {ttype} tags: {' '.join(tags_strings)}")

            hypo_joint_str = dictionary.string(
                joint_string,
                self.post_process,
                escape_unk=True,
                extra_symbols_to_ignore=symbols_to_ignore,
            )
            detok_hypo_str = self.decode_fn(hypo_joint_str)
            detok_hypo_str = move_tags_after_space(move_tags_to_start_or_end(detok_hypo_str))
        else:
            hypo_str = dictionary.string(
                tokens,
                self.post_process,
                escape_unk=True,
                extra_symbols_to_ignore=symbols_to_ignore,
            )
            detok_hypo_str = self.decode_fn(hypo_str)

        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Request ID[{request_id}] {ttype} hypo: {detok_hypo_str}")
        return detok_hypo_str
