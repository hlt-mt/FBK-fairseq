# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, List, Optional

import torch
from fairseq import search
from fairseq.models.fairseq_encoder import EncoderOut
from torch import Tensor

from fairseq.sequence_generator import SequenceGenerator, EnsembleModel


class HierarchicalBeamSearch(search.BeamSearch):
    def __init__(self, tgt_dict):
        super().__init__(tgt_dict)

    @torch.jit.export
    def step(self, step: int, lprobs, scores: Optional[Tensor], prev_scores: Tensor):
        bsz, beam_size, vocab_size = lprobs.size()

        if step == 0:
            # at the first step add the scores from previous beam search
            lprobs = lprobs + prev_scores
        else:
            # make probs contain cumulative scores for each hypothesis
            assert scores is not None
            lprobs = lprobs + scores[:, :, step - 1].unsqueeze(-1)

        top_prediction = torch.topk(
            lprobs.view(bsz, -1),
            k=min(
                # Take the best 2 x beam_size predictions. We'll choose the first
                # beam_size of these which don't predict eos to continue with.
                beam_size * 2,
                lprobs.view(bsz, -1).size(1) - 1,  # -1 so we never select pad
            ),
        )
        scores_buf = top_prediction[0]
        indices_buf = top_prediction[1]
        if torch.__version__ < '1.6.0':
            beams_buf = torch.div(indices_buf, vocab_size)
        else:
            beams_buf = torch.floor_divide(indices_buf, vocab_size)
        indices_buf = indices_buf.fmod(vocab_size)
        return scores_buf, indices_buf, beams_buf


class TwoPhaseSequenceGenerator(SequenceGenerator):
    def __init__(
        self,
        models,
        src_dict,
        tgt_dict,
        beam_size=1,
        max_len_a=0,
        max_len_b=200,
        min_len=1,
        normalize_scores=True,
        len_penalty=1.0,
        unk_penalty=0.0,
        temperature=1.0,
        match_source_len=False,
        no_repeat_ngram_size=0,
        search_strategy=None,
        eos=None,
    ):
        """Generates transcripts and translations of a given source audio.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models,
                currently support fairseq.models.TransformerModel for scripting
            beam_size (int, optional): beam width (default: 1)
            max_len_a/b (int, optional): generate sequences of maximum length
                ax + b, where x is the source length
            min_len (int, optional): the minimum length of the generated output
                (not including end-of-sentence)
            normalize_scores (bool, optional): normalize scores by the length
                of the output (default: True)
            len_penalty (float, optional): length penalty, where <1.0 favors
                shorter, >1.0 favors longer sentences (default: 1.0)
            unk_penalty (float, optional): unknown word penalty, where <0
                produces more unks, >0 produces fewer (default: 0.0)
            retain_dropout (bool, optional): use dropout when generating
                (default: False)
            temperature (float, optional): temperature, where values
                >1.0 produce more uniform samples and values <1.0 produce
                sharper samples (default: 1.0)
            match_source_len (bool, optional): outputs should match the source
                length (default: False)
        """
        super().__init__(
            models, tgt_dict,
            beam_size=beam_size,
            max_len_a=max_len_a,
            max_len_b=max_len_b,
            min_len=min_len,
            normalize_scores=normalize_scores,
            len_penalty=len_penalty,
            unk_penalty=unk_penalty,
            temperature=temperature,
            match_source_len=match_source_len,
            no_repeat_ngram_size=no_repeat_ngram_size,
            search_strategy=search_strategy,
            eos=eos
        )
        if isinstance(models, EnsembleTwoPhaseModel):
            self.model = models
        else:
            self.model = EnsembleTwoPhaseModel(models)
        self.src_pad = src_dict.pad()
        self.src_unk = src_dict.unk()
        self.src_eos = src_dict.eos() if eos is None else eos
        self.src_vocab_size = len(src_dict)
        self.src_search = search.BeamSearch(src_dict)
        self.search = HierarchicalBeamSearch(tgt_dict)

    def cuda(self):
        self.model.cuda()
        return self

    def _generate(
            self,
            sample: Dict[str, Dict[str, Tensor]],
            prefix_tokens: Optional[Tensor] = None,
            constraints: Optional[Tensor] = None,
            bos_token: Optional[int] = None,
    ):
        net_input = sample["net_input"]
        # TODO: should not use audio features...
        src_tokens = net_input["src_tokens"]
        # length of the source text being the character length except EndOfSentence and pad
        src_lengths = (
            (src_tokens.ne(self.src_eos) & src_tokens.ne(self.src_pad)).long().sum(dim=1)
        )
        # bsz: total number of sentences in beam
        input_size = src_tokens.size()
        bsz, src_len = input_size[0], input_size[1]
        beam_size = self.beam_size

        max_len: int = -1
        if self.match_source_len:
            max_len = src_lengths.max().item()
        else:
            max_len = min(
                int(self.max_len_a * src_len + self.max_len_b),
                # exclude the EOS marker
                self.model.max_decoder_positions() - 1,
            )
        assert (
                self.min_len <= max_len
        ), "min_len cannot be larger than max_len, please adjust these!"
        # compute the encoder output for each beam
        encoder_outs = self.model.forward_encoder(net_input)

        # placeholder of indices for bsz * beam_size to hold tokens and accumulative scores
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()
        encoder_outs = self.model.reorder_encoder_out(encoder_outs, new_order)
        # ensure encoder_outs is a List.
        assert encoder_outs is not None
        aux_nbest = self._generate_aux(
            sample, encoder_outs, prefix_tokens=prefix_tokens, bos_token=bos_token)
        return self._generate_tgt(
            aux_nbest, encoder_outs, prefix_tokens=prefix_tokens, bos_token=bos_token)

    def _generate_tgt(
        self,
        aux_nbest: List[List[Dict[str, Tensor]]],
        encoder_outs: List[EncoderOut],
        prefix_tokens: Optional[Tensor] = None,
        bos_token: Optional[int] = None,
    ):
        incremental_states = torch.jit.annotate(
            List[Dict[str, Dict[str, Optional[Tensor]]]],
            [
                torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {})
                for i in range(self.model.models_size)
            ],
        )
        # bsz: total number of sentences in beam
        bsz = len(aux_nbest)
        beam_size = self.beam_size
        max_aux_len = max([cand["tokens"].shape[0] for sent in aux_nbest for cand in sent])
        src_tokens = (
            torch.zeros(bsz, beam_size, max_aux_len).long().fill_(self.src_pad).to(
                aux_nbest[0][0]["tokens"].device)
        )
        src_tags = (
            torch.zeros(bsz, beam_size, max_aux_len).long().to(
                aux_nbest[0][0]["tokens"].device)
        )
        for i_batch in range(len(aux_nbest)):
            for i_best in range(len(aux_nbest[i_batch])):
                cand = aux_nbest[i_batch][i_best]
                src_tokens[i_batch, i_best, :cand["tokens"].shape[0]] = cand["tokens"]
                if cand["aux_tags"] is not None:
                    src_tags[i_batch, i_best, :cand["aux_tags"].shape[0]] = cand["aux_tags"]
                else:
                    src_tags = None

        src_tokens = src_tokens.view(bsz * beam_size, -1)
        if src_tags is not None:
            src_tags = src_tags.view(bsz * beam_size, -1)
        # length of the source text being the character length except EndOfSentence and pad
        src_lengths = (
            (src_tokens.ne(self.src_eos) & src_tokens.ne(self.src_pad)).long().sum(dim=1)
        )
        src_len = src_tokens.size()[1]

        auxiliary_outputs = torch.zeros(
            bsz, beam_size, max_aux_len, aux_nbest[0][0]["auxiliary_out"].shape[1]).float().fill_(
            self.src_pad).to(src_tokens.device)
        for i_batch in range(len(aux_nbest)):
            for i_best in range(len(aux_nbest[i_batch])):
                cand = aux_nbest[i_batch][i_best]
                auxiliary_outputs[i_batch, i_best, :cand["auxiliary_out"].shape[0], :] = cand["auxiliary_out"]

        auxiliary_outputs = auxiliary_outputs.view(bsz * beam_size, max_aux_len, -1)

        prev_scores = torch.stack(
            [cand["score"] for sent in aux_nbest for cand in sent]).view(bsz, beam_size, 1)

        if self.match_source_len:
            max_len = src_lengths.max().item()
        else:
            max_len = min(
                int(self.max_len_a * src_len + self.max_len_b),
                # exclude the EOS marker
                self.model.max_decoder_positions() - 1,
            )
        assert (
            self.min_len <= max_len
        ), "min_len cannot be larger than max_len, please adjust these!"
        # compute the encoder output for each beam

        # initialize buffers
        scores = (
            torch.zeros(bsz * beam_size, max_len + 1).to(src_tokens).float()
        )  # +1 for eos; pad is never choosed for scoring

        tokens = (
            torch.zeros(bsz * beam_size, max_len + 2)
            .to(src_tokens)
            .long()
            .fill_(self.pad)
        )  # +2 for eos and pad
        tokens[:, 0] = self.eos if bos_token is None else bos_token
        attn: Optional[Tensor] = None
        tags: Optional[Tensor] = None

        # The blacklist indicates candidates that should be ignored.
        # For example, suppose we're sampling and have already finalized 2/5
        # samples. Then the blacklist would mark 2 positions as being ignored,
        # so that we only finalize the remaining 3 samples.
        blacklist = (
            torch.zeros(bsz, beam_size).to(src_tokens).eq(-1)
        )  # forward and backward-compatible False mask

        # list of completed sentences
        finalized = torch.jit.annotate(
            List[List[Dict[str, Tensor]]],
            [torch.jit.annotate(List[Dict[str, Tensor]], []) for i in range(bsz)],
        )  # contains lists of dictionaries of infomation about the hypothesis being finalized at each step

        finished = [
            False for i in range(bsz)
        ]  # a boolean array indicating if the sentence at the index is finished or not
        num_remaining_sent = bsz  # number of sentences remaining

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (torch.arange(0, bsz) * beam_size).unsqueeze(1).type_as(tokens)
        cand_offsets = torch.arange(0, cand_size).type_as(tokens)

        reorder_state: Optional[Tensor] = None
        batch_idxs: Optional[Tensor] = None
        for step in range(max_len + 1):  # one extra step for EOS marker
            # reorder decoder internal states based on the prev choice of beams
            # print(f'step: {step}')
            if reorder_state is not None:
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(
                        batch_idxs
                    )
                    reorder_state.view(-1, beam_size).add_(
                        corr.unsqueeze(-1) * beam_size
                    )
                self.model.reorder_incremental_state(incremental_states, reorder_state)
                encoder_outs = self.model.reorder_encoder_out(
                    encoder_outs, reorder_state
                )
                src_tokens = src_tokens.index_select(0, reorder_state)
                if src_tags is not None:
                    src_tags = src_tags.index_select(0, reorder_state)
                prev_scores = prev_scores.view(-1).index_select(0, reorder_state).view(-1, beam_size, 1)
                auxiliary_outputs = auxiliary_outputs.index_select(0, reorder_state)

            lprobs, avg_attn_scores, tags_lprobs = self.model.forward_decoder(
                tokens[:, : step + 1],
                encoder_outs,
                incremental_states,
                src_tokens,
                auxiliary_outputs,
                self.temperature
            )
            lprobs[lprobs != lprobs] = torch.tensor(-math.inf).to(lprobs)

            lprobs[:, self.pad] = -math.inf  # never select pad
            lprobs[:, self.unk] -= self.unk_penalty  # apply unk penalty

            # handle max length constraint
            if step >= max_len:
                lprobs[:, : self.eos] = -math.inf
                lprobs[:, self.eos + 1 :] = -math.inf

            # handle prefix tokens (possibly with different lengths)
            if (
                prefix_tokens is not None
                and step < prefix_tokens.size(1)
                and step < max_len
            ):
                lprobs, tokens, scores = self._prefix_tokens(
                    step, lprobs, scores, tokens, prefix_tokens, beam_size, self.pad, self.eos
                )
            elif step < self.min_len:
                # minimum length constraint (does not apply if using prefix_tokens)
                lprobs[:, self.eos] = -math.inf

            # Record attention scores, only support avg_attn_scores is a Tensor
            if avg_attn_scores is not None:
                if attn is None:
                    attn = torch.empty(
                        bsz * beam_size, avg_attn_scores.size(1), max_len + 2
                    ).to(scores)
                attn[:, :, step + 1].copy_(avg_attn_scores)

            # Record tags, only support avg_attn_scores is a Tensor
            if tags_lprobs is not None:
                if tags is None:
                    tags = torch.empty(bsz * beam_size, max_len + 2).to(scores)
                tags[:, step + 1] = torch.argmax(tags_lprobs, dim=-1)

            scores = scores.type_as(lprobs)
            eos_bbsz_idx = torch.empty(0).to(
                tokens
            )  # indices of hypothesis ending with eos (finished sentences)
            eos_scores = torch.empty(0).to(
                scores
            )  # scores of hypothesis ending with eos (finished sentences)

            self.search.set_src_lengths(src_lengths)

            if self.no_repeat_ngram_size > 0:
                lprobs = self._no_repeat_ngram(tokens, lprobs, bsz, beam_size, step)

            cand_scores, cand_indices, cand_beams = self.search.step(
                step,
                lprobs.view(bsz, -1, self.vocab_size),
                scores.view(bsz, beam_size, -1)[:, :, :step],
                prev_scores,
            )

            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)

            # finalize hypotheses that end in eos
            eos_mask = cand_indices.eq(self.eos) & cand_scores.ne(-math.inf)
            eos_mask[:, :beam_size][blacklist] = torch.tensor(0).to(eos_mask)

            # only consider eos when it's among the top beam_size indices
            eos_bbsz_idx = torch.masked_select(
                cand_bbsz_idx[:, :beam_size], mask=eos_mask[:, :beam_size]
            )

            finalized_sents: List[int] = []
            if eos_bbsz_idx.numel() > 0:
                eos_scores = torch.masked_select(
                    cand_scores[:, :beam_size], mask=eos_mask[:, :beam_size]
                )
                finalized_sents = self.finalize_hypos(
                    step,
                    eos_bbsz_idx,
                    eos_scores,
                    tokens,
                    src_tokens,
                    src_tags,
                    scores,
                    finalized,
                    finished,
                    beam_size,
                    attn,
                    tags,
                    src_lengths,
                    max_len,
                )
                num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            assert step < max_len

            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)

                # construct batch_idxs which holds indices of batches to keep for the next pass
                batch_mask = torch.ones(bsz).to(cand_indices)
                batch_mask[
                    torch.tensor(finalized_sents).to(cand_indices)
                ] = torch.tensor(0).to(batch_mask)
                batch_idxs = batch_mask.nonzero().squeeze(-1)

                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]

                if prefix_tokens is not None:
                    prefix_tokens = prefix_tokens[batch_idxs]
                src_lengths = src_lengths[batch_idxs]
                blacklist = blacklist[batch_idxs]

                scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                if attn is not None:
                    attn = attn.view(bsz, -1)[batch_idxs].view(
                        new_bsz * beam_size, attn.size(1), -1
                    )
                if tags is not None:
                    tags = tags.view(bsz, -1)[batch_idxs].view(
                        new_bsz * beam_size, -1
                    )
                bsz = new_bsz
            else:
                batch_idxs = None
            # set active_mask so that values > cand_size indicate eos hypos
            # and values < cand_size indicate candidate active hypos.
            # After, the min values per row are the top candidate active hypos

            # Rewrite the operator since the element wise or is not supported in torchscript.

            eos_mask[:, :beam_size] = ~((~blacklist) & (~eos_mask[:, :beam_size]))
            active_mask = torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[: eos_mask.size(1)],
            )

            # get the top beam_size active hypotheses, which are just the hypos
            # with the smallest values in active_mask
            new_blacklist, active_hypos = torch.topk(
                active_mask, k=beam_size, dim=1, largest=False
            )

            # update blacklist to ignore any finalized hypos
            blacklist = new_blacklist.ge(cand_size)[:, :beam_size]
            assert (~blacklist).any(dim=1).all()

            active_bbsz_idx = torch.gather(cand_bbsz_idx, dim=1, index=active_hypos)
            active_scores = torch.gather(cand_scores, dim=1, index=active_hypos)

            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)

            # copy tokens and scores for active hypotheses
            tokens[:, : step + 1] = torch.index_select(
                tokens[:, : step + 1], dim=0, index=active_bbsz_idx
            )
            tokens.view(bsz, beam_size, -1)[:, :, step + 1] = torch.gather(
                cand_indices, dim=1, index=active_hypos
            )
            if step > 0:
                scores[:, :step] = torch.index_select(
                    scores[:, :step], dim=0, index=active_bbsz_idx
                )
            scores.view(bsz, beam_size, -1)[:, :, step] = torch.gather(
                cand_scores, dim=1, index=active_hypos
            )

            # copy attention for active hypotheses
            if attn is not None:
                attn[:, :, : step + 2] = torch.index_select(
                    attn[:, :, : step + 2], dim=0, index=active_bbsz_idx
                )
            if tags is not None:
                tags[:, : step + 2] = torch.index_select(
                    tags[:, : step + 2], dim=0, index=active_bbsz_idx
                )

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx

        # sort by score descending
        for sent in range(len(finalized)):
            scores = torch.tensor(
                [float(elem["score"].item()) for elem in finalized[sent]]
            )
            _, sorted_scores_indices = torch.sort(scores, descending=True)
            finalized[sent] = [finalized[sent][ssi] for ssi in sorted_scores_indices]
            finalized[sent] = torch.jit.annotate(
                List[Dict[str, Tensor]], finalized[sent]
            )

        return finalized

    def _generate_aux(
        self,
        sample: Dict[str, Dict[str, Tensor]],
        encoder_outs: List[EncoderOut],
        prefix_tokens: Optional[Tensor] = None,
        bos_token: Optional[int] = None,
    ):
        auxiliary_incremental_states = torch.jit.annotate(
            List[Dict[str, Dict[str, Optional[Tensor]]]],
            [
                torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {})
                for _ in range(self.model.models_size)
            ],
        )
        net_input = sample["net_input"]
        # TODO: should not use audio features...
        src_tokens = net_input["src_tokens"]
        # length of the source text being the character length except EndOfSentence and pad
        src_lengths = (
            (src_tokens.ne(self.src_eos) & src_tokens.ne(self.src_pad)).long().sum(dim=1)
        )
        # bsz: total number of sentences in beam
        input_size = src_tokens.size()
        bsz, src_len = input_size[0], input_size[1]
        beam_size = self.beam_size
        max_len: int = -1
        if self.match_source_len:
            max_len = src_lengths.max().item()
        else:
            max_len = min(
                int(self.max_len_a * src_len + self.max_len_b),
                # exclude the EOS marker
                self.model.max_decoder_positions() - 1,
            )
        # initialize buffers
        scores = (
            torch.zeros(bsz * beam_size, max_len + 1).to(src_tokens).float()
        )  # +1 for eos; pad is never chosen for scoring
        aux_tokens = (
            torch.zeros(bsz * beam_size, max_len + 2)
            .to(src_tokens)
            .long()
            .fill_(self.src_pad)
        )  # +2 for eos and pad
        aux_tokens[:, 0] = self.src_eos if bos_token is None else bos_token
        attn: Optional[Tensor] = None
        aux_tags: Optional[Tensor] = None

        # The ignorelist indicates candidates that should be ignored.
        # For example, suppose we're sampling and have already finalized 2/5
        # samples. Then the ignorelist would mark 2 positions as being ignored,
        # so that we only finalize the remaining 3 samples.
        ignorelist = (
            torch.zeros(bsz, beam_size).to(src_tokens).eq(-1)
        )  # forward and backward-compatible False mask

        # list of completed sentences
        finalized = torch.jit.annotate(
            List[List[Dict[str, Tensor]]],
            [torch.jit.annotate(List[Dict[str, Tensor]], []) for i in range(bsz)],
        )  # contains lists of dictionaries of information about the hypothesis being finalized at each step

        finished = [
            False for i in range(bsz)
        ]  # a boolean array indicating if the sentence at the index is finished or not
        num_remaining_sent = bsz  # number of sentences remaining

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (torch.arange(0, bsz) * beam_size).unsqueeze(1).type_as(aux_tokens)
        cand_offsets = torch.arange(0, cand_size).type_as(aux_tokens)

        reorder_state: Optional[Tensor] = None
        batch_idxs: Optional[Tensor] = None
        aux_outputs: Optional[Tensor] = None
        for step in range(max_len + 1):  # one extra step for EOS marker
            # reorder decoder internal states based on the prev choice of beams
            # print(f'step: {step}')
            if reorder_state is not None:
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(
                        batch_idxs
                    )
                    reorder_state.view(-1, beam_size).add_(
                        corr.unsqueeze(-1) * beam_size
                    )
                self.model.reorder_auxiliary_incremental_state(auxiliary_incremental_states, reorder_state)
                encoder_outs = self.model.reorder_encoder_out(
                    encoder_outs, reorder_state
                )
                aux_outputs = aux_outputs.index_select(1, reorder_state)

            lprobs, aux_out, avg_attn_scores, aux_tags_lprobs = self.model.forward_auxiliary_decoder(
                aux_tokens[:, : step + 1], encoder_outs, auxiliary_incremental_states, self.temperature
            )
            if step == 0:
                # We need to initialize this here as we don't know the last dimension (C)
                # until we do the first step
                aux_outputs = (
                    torch.zeros(max_len + 1, bsz * beam_size, aux_out.shape[-1]).to(src_tokens).float()
                )
            # Assign the auxiliary outputs for this decoding step (only the current decoding step is returned)
            aux_outputs[step] = aux_out.squeeze(1)
            lprobs[lprobs != lprobs] = torch.tensor(-math.inf).to(lprobs)

            lprobs[:, self.src_pad] = -math.inf  # never select pad
            lprobs[:, self.src_unk] -= self.unk_penalty  # apply unk penalty

            # handle max length constraint
            if step >= max_len:
                lprobs[:, : self.src_eos] = -math.inf
                lprobs[:, self.src_eos + 1 :] = -math.inf

            # handle prefix tokens (possibly with different lengths)
            if (
                prefix_tokens is not None
                and step < prefix_tokens.size(1)
                and step < max_len
            ):
                lprobs, aux_tokens, scores = self._prefix_tokens(
                    step, lprobs, scores, aux_tokens, prefix_tokens, beam_size, self.src_pad, self.src_eos
                )
            elif step < self.min_len:
                # minimum length constraint (does not apply if using prefix_tokens)
                lprobs[:, self.src_eos] = -math.inf

            # Record attention scores, only support avg_attn_scores is a Tensor
            if avg_attn_scores is not None:
                if attn is None:
                    attn = torch.empty(
                        bsz * beam_size, avg_attn_scores.size(1), max_len + 2
                    ).to(scores)
                attn[:, :, step + 1].copy_(avg_attn_scores)

            # Record tags
            if aux_tags_lprobs is not None:
                if aux_tags is None:
                    aux_tags = torch.empty(bsz * beam_size, max_len + 2).to(scores)
                aux_tags[:, step + 1] = torch.argmax(aux_tags_lprobs, dim=-1)

            scores = scores.type_as(lprobs)
            eos_bbsz_idx = torch.empty(0).to(
                aux_tokens
            )  # indices of hypothesis ending with eos (finished sentences)
            eos_scores = torch.empty(0).to(
                scores
            )  # scores of hypothesis ending with eos (finished sentences)

            self.src_search.set_src_lengths(src_lengths)

            if self.no_repeat_ngram_size > 0:
                lprobs = self._no_repeat_ngram(aux_tokens, lprobs, bsz, beam_size, step)

            cand_scores, cand_indices, cand_beams = self.src_search.step(
                step,
                lprobs.view(bsz, -1, self.src_vocab_size),
                scores.view(bsz, beam_size, -1)[:, :, :step],
            )

            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)

            # finalize hypotheses that end in eos
            eos_mask = cand_indices.eq(self.src_eos) & cand_scores.ne(-math.inf)
            eos_mask[:, :beam_size][ignorelist] = torch.tensor(0).to(eos_mask)

            # only consider eos when it's among the top beam_size indices
            eos_bbsz_idx = torch.masked_select(
                cand_bbsz_idx[:, :beam_size], mask=eos_mask[:, :beam_size]
            )

            finalized_sents: List[int] = []
            if eos_bbsz_idx.numel() > 0:
                eos_scores = torch.masked_select(
                    cand_scores[:, :beam_size], mask=eos_mask[:, :beam_size]
                )
                finalized_sents = self.finalize_aux_hypos(
                    step,
                    eos_bbsz_idx,
                    eos_scores,
                    aux_tokens,
                    encoder_outs,
                    aux_outputs,
                    scores,
                    finalized,
                    finished,
                    beam_size,
                    attn,
                    aux_tags,
                    src_lengths,
                    max_len,
                    self.src_eos,
                )
                num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            assert step < max_len

            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)

                # construct batch_idxs which holds indices of batches to keep for the next pass
                batch_mask = torch.ones(bsz).to(cand_indices)
                batch_mask[
                    torch.tensor(finalized_sents).to(cand_indices)
                ] = torch.tensor(0).to(batch_mask)
                batch_idxs = batch_mask.nonzero().squeeze(-1)

                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]

                if prefix_tokens is not None:
                    prefix_tokens = prefix_tokens[batch_idxs]
                src_lengths = src_lengths[batch_idxs]
                ignorelist = ignorelist[batch_idxs]

                scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                aux_tokens = aux_tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                if aux_tags is not None:
                    aux_tags = aux_tags.view(bsz, -1)[batch_idxs].view(
                        new_bsz * beam_size, -1
                    )
                if attn is not None:
                    attn = attn.view(bsz, -1)[batch_idxs].view(
                        new_bsz * beam_size, attn.size(1), -1
                    )
                bsz = new_bsz
            else:
                batch_idxs = None
            # set active_mask so that values > cand_size indicate eos hypos
            # and values < cand_size indicate candidate active hypos.
            # After, the min values per row are the top candidate active hypos

            # Rewrite the operator since the element wise or is not supported in torchscript.

            eos_mask[:, :beam_size] = ~((~ignorelist) & (~eos_mask[:, :beam_size]))
            active_mask = torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[: eos_mask.size(1)],
            )

            # get the top beam_size active hypotheses, which are just the hypos
            # with the smallest values in active_mask
            new_ignorelist, active_hypos = torch.topk(
                active_mask, k=beam_size, dim=1, largest=False
            )

            # update ignorelist to ignore any finalized hypos
            ignorelist = new_ignorelist.ge(cand_size)[:, :beam_size]
            assert (~ignorelist).any(dim=1).all()

            active_bbsz_idx = torch.gather(cand_bbsz_idx, dim=1, index=active_hypos)
            active_scores = torch.gather(cand_scores, dim=1, index=active_hypos)

            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)

            # copy tokens and scores for active hypotheses
            aux_tokens[:, : step + 1] = torch.index_select(
                aux_tokens[:, : step + 1], dim=0, index=active_bbsz_idx
            )
            aux_tokens.view(bsz, beam_size, -1)[:, :, step + 1] = torch.gather(
                cand_indices, dim=1, index=active_hypos
            )
            if step > 0:
                scores[:, :step] = torch.index_select(
                    scores[:, :step], dim=0, index=active_bbsz_idx
                )
            scores.view(bsz, beam_size, -1)[:, :, step] = torch.gather(
                cand_scores, dim=1, index=active_hypos
            )

            # copy attention for active hypotheses
            if attn is not None:
                attn[:, :, : step + 2] = torch.index_select(
                    attn[:, :, : step + 2], dim=0, index=active_bbsz_idx
                )

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx

        # sort by score descending
        # for sent in range(len(finalized)):
        #     # make into beam container
        #     BCList = [
        #         BeamContainer(elem["score"].item(), elem) for elem in finalized[sent]
        #     ]
        #     BCList.sort()
        #     BCList.reverse()
        #     finalized[sent] = torch.jit.annotate(
        #         List[Dict[str, Tensor]], [x.elem for x in BCList]
        #     )

        return finalized

    def _prefix_tokens(
        self, step: int, lprobs, scores, tokens, prefix_tokens, beam_size: int, pad, eos
    ):
        """Handle prefix tokens"""
        prefix_toks = prefix_tokens[:, step].unsqueeze(-1).repeat(1, beam_size).view(-1)
        prefix_lprobs = lprobs.gather(-1, prefix_toks.unsqueeze(-1))
        prefix_mask = prefix_toks.ne(pad)
        lprobs[prefix_mask] = torch.tensor(-math.inf).to(lprobs)
        lprobs[prefix_mask] = lprobs[prefix_mask].scatter(
            -1, prefix_toks[prefix_mask].unsqueeze(-1), prefix_lprobs[prefix_mask]
        )
        # if prefix includes eos, then we should make sure tokens and
        # scores are the same across all beams
        eos_mask = prefix_toks.eq(eos)
        if eos_mask.any():
            # validate that the first beam matches the prefix
            first_beam = tokens[eos_mask].view(-1, beam_size, tokens.size(-1))[
                :, 0, 1 : step + 1
            ]
            eos_mask_batch_dim = eos_mask.view(-1, beam_size)[:, 0]
            target_prefix = prefix_tokens[eos_mask_batch_dim][:, :step]
            assert (first_beam == target_prefix).all()

            # copy tokens, scores and lprobs from the first beam to all beams
            tokens = self.replicate_first_beam(tokens, eos_mask_batch_dim, beam_size)
            scores = self.replicate_first_beam(scores, eos_mask_batch_dim, beam_size)
            lprobs = self.replicate_first_beam(lprobs, eos_mask_batch_dim, beam_size)
        return lprobs, tokens, scores

    def finalize_aux_hypos(
        self,
        step: int,
        bbsz_idx,
        eos_scores,
        tokens,
        encoder_outs,
        decoder_out,
        scores,
        finalized: List[List[Dict[str, Tensor]]],
        finished: List[bool],
        beam_size: int,
        attn: Optional[Tensor],
        aux_tags: Optional[Tensor],
        src_lengths,
        max_len: int,
        eos,
    ):
        """Finalize hypothesis, store finalized information in `finalized`, and change `finished` accordingly.
        Returns number of sentences being finalized.
        Args:
            bbsz_idx (Tensor):
        """
        assert bbsz_idx.numel() == eos_scores.numel()

        # clone relevant token and attention tensors
        tokens_clone = tokens.index_select(0, bbsz_idx)[
            :, 1 : step + 2
        ]  # skip the first index, which is EOS
        decoder_out_clone = decoder_out.index_select(1, bbsz_idx)[: step + 1].transpose(0, 1)
        encoder_outs_clone = self.model.reorder_encoder_out(encoder_outs, bbsz_idx)

        tokens_clone[:, step] = eos
        attn_clone = (
            attn.index_select(0, bbsz_idx)[:, :, 1 : step + 2]
            if attn is not None
            else None
        )
        aux_tags_clone = (
            aux_tags.index_select(0, bbsz_idx)[:, 1: step + 2]
            if aux_tags is not None
            else None
        )

        # compute scores per token position
        pos_scores = scores.index_select(0, bbsz_idx)[:, : step + 1]
        pos_scores[:, step] = eos_scores
        # convert from cumulative to per-position scores
        pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

        # normalize sentence-level scores
        if self.normalize_scores:
            eos_scores /= (step + 1) ** self.len_penalty

        cum_unfin: List[int] = []
        prev = 0
        for f in finished:
            if f:
                prev += 1
            else:
                cum_unfin.append(prev)

        # set() is not supported in script export
        sents_seen: Dict[str, Optional[Tensor]] = {}
        for i in range(bbsz_idx.size()[0]):
            idx = bbsz_idx[i]
            score = eos_scores[i]
            unfin_idx = idx // beam_size
            sent = unfin_idx + cum_unfin[unfin_idx]
            # Cannot create dict for key type '(int, int)' in torchscript.
            # The workaround is to cast int to string
            seen = str(sent.item()) + "_" + str(unfin_idx.item())
            if seen not in sents_seen:
                sents_seen[seen] = None

            if self.match_source_len and step > src_lengths[unfin_idx]:
                score = torch.tensor(-math.inf).to(score)

            if len(finalized[sent]) < beam_size:
                if attn_clone is not None:
                    # remove padding tokens from attn scores
                    hypo_attn = attn_clone[i]
                else:
                    hypo_attn = torch.empty(0)
                if aux_tags_clone is not None:
                    aux_hypo_tags = aux_tags_clone[i]
                else:
                    aux_hypo_tags = None
                finalized[sent].append(
                    {
                        "tokens": tokens_clone[i],
                        "auxiliary_out": decoder_out_clone[i],
                        "encoder_outs": self.model.reorder_encoder_out(
                            encoder_outs_clone, torch.tensor([i], dtype=torch.long).to(tokens_clone.device)),
                        "score": score,
                        "attention": hypo_attn,  # src_len x tgt_len
                        "alignment": torch.empty(0),
                        "positional_scores": pos_scores[i],
                        "aux_tags": aux_hypo_tags,
                    }
                )

        newly_finished: List[int] = []
        for seen in sents_seen.keys():
            # check termination conditions for this sentence
            sent: int = int(float(seen.split("_")[0]))
            unfin_idx: int = int(float(seen.split("_")[1]))
            if not finished[sent] and self.is_finished(
                step, unfin_idx, max_len, len(finalized[sent]), beam_size
            ):
                finished[sent] = True
                newly_finished.append(unfin_idx)
        return newly_finished

    def finalize_hypos(
        self,
        step: int,
        bbsz_idx,
        eos_scores,
        tokens,
        src_tokens,
        src_tags,
        scores,
        finalized: List[List[Dict[str, Tensor]]],
        finished: List[bool],
        beam_size: int,
        attn: Optional[Tensor],
        tags: Optional[Tensor],
        src_lengths,
        max_len: int,
    ):
        """Finalize hypothesis, store finalized information in `finalized`, and change `finished` accordingly.
        Returns number of sentences being finalized.
        Args:
            bbsz_idx (Tensor):
        """
        assert bbsz_idx.numel() == eos_scores.numel()

        # clone relevant token and attention tensors
        tokens_clone = tokens.index_select(0, bbsz_idx)[
            :, 1 : step + 2
        ]  # skip the first index, which is EOS
        src_tokens_clone = src_tokens.index_select(0, bbsz_idx)
        src_tags_clone = None
        if src_tags is not None:
            src_tags_clone = src_tags.index_select(0, bbsz_idx)

        tokens_clone[:, step] = self.eos
        attn_clone = (
            attn.index_select(0, bbsz_idx)[:, :, 1 : step + 2]
            if attn is not None
            else None
        )

        tags_clone = (
            tags.index_select(0, bbsz_idx)[:, 1: step + 2]
            if tags is not None
            else None
        )

        # compute scores per token position
        pos_scores = scores.index_select(0, bbsz_idx)[:, : step + 1]
        pos_scores[:, step] = eos_scores
        # convert from cumulative to per-position scores
        pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

        # normalize sentence-level scores
        if self.normalize_scores:
            eos_scores /= (step + 1) ** self.len_penalty

        cum_unfin: List[int] = []
        prev = 0
        for f in finished:
            if f:
                prev += 1
            else:
                cum_unfin.append(prev)

        # set() is not supported in script export
        sents_seen: Dict[str, Optional[Tensor]] = {}
        for i in range(bbsz_idx.size()[0]):
            idx = bbsz_idx[i]
            score = eos_scores[i]
            unfin_idx = idx // beam_size
            sent = unfin_idx + cum_unfin[unfin_idx]
            # Cannot create dict for key type '(int, int)' in torchscript.
            # The workaround is to cast int to string
            seen = str(sent.item()) + "_" + str(unfin_idx.item())
            if seen not in sents_seen:
                sents_seen[seen] = None

            if self.match_source_len and step > src_lengths[unfin_idx]:
                score = torch.tensor(-math.inf).to(score)

            if len(finalized[sent]) < beam_size:
                if attn_clone is not None:
                    # remove padding tokens from attn scores
                    hypo_attn = attn_clone[i]
                else:
                    hypo_attn = torch.empty(0)

                if tags_clone is not None:
                    hypo_tags = tags_clone[i]
                else:
                    hypo_tags = None

                src_mask = src_tokens_clone[i] != self.src_pad
                aux_tags = None
                if src_tags_clone is not None:
                    aux_tags = src_tags_clone[i].masked_select(src_mask)
                finalized[sent].append(
                    {
                        "tokens": tokens_clone[i],
                        "score": score,
                        "aux_tokens": src_tokens_clone[i].masked_select(src_mask),
                        "attention": hypo_attn,  # src_len x tgt_len
                        "alignment": torch.empty(0),
                        "positional_scores": pos_scores[i],
                        "tags": hypo_tags,
                        "aux_tags": aux_tags,
                    }
                )

        newly_finished: List[int] = []
        for seen in sents_seen.keys():
            # check termination conditions for this sentence
            sent: int = int(float(seen.split("_")[0]))
            unfin_idx: int = int(float(seen.split("_")[1]))
            if not finished[sent] and self.is_finished(
                step, unfin_idx, max_len, len(finalized[sent]), beam_size
            ):
                finished[sent] = True
                newly_finished.append(unfin_idx)
        return newly_finished


class EnsembleTwoPhaseModel(EnsembleModel):
    """A wrapper around an ensemble of models."""

    @torch.jit.export
    def forward_decoder(
            self, tokens,
            encoder_outs: List[EncoderOut],
            incremental_states,
            aux_tokens: Tensor,
            aux_decoder_out: Tensor,
            temperature: float = 1.0
    ):
        log_probs = []
        tags_lprobs = []
        avg_tags_lprobs: Optional[Tensor] = None
        avg_attn: Optional[Tensor] = None
        encoder_out: Optional[EncoderOut] = None
        for i, model in enumerate(self.models):
            if self.has_encoder():
                encoder_out = encoder_outs[i]
            # decode each model
            if self.has_incremental_states():
                decoder_out = model.forward_decoder(
                    tokens,
                    auxiliary_out=aux_decoder_out,
                    auxiliary_tokens=aux_tokens,
                    encoder_out=encoder_out,
                    incremental_state=incremental_states[i],
                )
            else:
                decoder_out = model.forward_decoder(
                    tokens,
                    auxiliary_out=aux_decoder_out,
                    auxiliary_tokens=aux_tokens,
                    encoder_out=encoder_out)

            attn: Optional[Tensor] = None
            tags: Optional[Tensor] = None
            decoder_len = len(decoder_out)
            if decoder_len > 1 and decoder_out[1] is not None \
                    and isinstance(decoder_out[1], dict) and "tags" in decoder_out[1]:
                tags = torch.log_softmax(decoder_out[1]["tags"], dim=-1)
                tags = tags[:, -1, :]

            if decoder_len > 1 and decoder_out[1] is not None:
                if isinstance(decoder_out[1], Tensor):
                    attn = decoder_out[1]
                else:
                    attn_holder = decoder_out[1]["attn"]
                    if isinstance(attn_holder, Tensor):
                        attn = attn_holder
                    elif attn_holder is not None:
                        attn = attn_holder[0]
                if attn is not None:
                    attn = attn[:, -1, :]

            decoder_out_tuple = (
                decoder_out[0][:, -1:, :].div_(temperature),
                None if decoder_len <= 1 else decoder_out[1],
            )

            probs = model.get_normalized_probs(
                decoder_out_tuple, log_probs=True, sample=None
            )
            probs = probs[:, -1, :]
            if self.models_size == 1:
                return probs, attn, tags

            log_probs.append(probs)
            if tags is not None:
                tags_lprobs.append(tags)
            if attn is not None:
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)
        avg_probs = torch.logsumexp(torch.stack(log_probs, dim=0), dim=0) - math.log(
            self.models_size
        )
        if len(tags_lprobs) > 0:
            avg_tags_lprobs = torch.logsumexp(torch.stack(tags_lprobs, dim=0), dim=0) - math.log(
                self.models_size
            )
        if avg_attn is not None:
            avg_attn.div_(self.models_size)
        return avg_probs, avg_attn, avg_tags_lprobs

    @torch.jit.export
    def reorder_auxiliary_incremental_state(self, auxiliary_incremental_states, new_order):
        if not self.has_incremental_states():
            return
        for i, model in enumerate(self.models):
            model.auxiliary_decoder.reorder_incremental_state_scripting(
                auxiliary_incremental_states[i], new_order
            )

    @torch.jit.export
    def forward_auxiliary_decoder(
        self, tokens, encoder_outs: List[EncoderOut], auxiliary_incremental_states, temperature: float = 1.0
    ):
        log_probs = []
        aux_tags_lprobs = []
        avg_aux_tags_lprobs: Optional[Tensor] = None
        outs = []
        avg_attn: Optional[Tensor] = None
        encoder_out: Optional[EncoderOut] = None
        for i, model in enumerate(self.models):
            if self.has_encoder():
                encoder_out = encoder_outs[i]
            # decode each model
            if self.has_incremental_states():
                decoder_out = model.auxiliary_decoder.forward(
                    tokens,
                    encoder_out=encoder_out,
                    incremental_state=auxiliary_incremental_states[i],
                    features_only=True,
                )
            else:
                decoder_out = model.auxiliary_decoder.forward(
                    tokens, encoder_out=encoder_out, features_only=True)

            decoder_out_emb = decoder_out[0]
            decoder_out = (model.auxiliary_decoder.output_layer(decoder_out[0]), decoder_out[1])
            attn: Optional[Tensor] = None
            aux_tags: Optional[Tensor] = None
            decoder_len = len(decoder_out)
            if decoder_len > 1 and decoder_out[1] is not None:
                if isinstance(decoder_out[1], dict) and "tags" in decoder_out[1]:
                    aux_tags = torch.log_softmax(decoder_out[1]["tags"], dim=-1)
                    aux_tags = aux_tags[:, -1, :]
                if isinstance(decoder_out[1], Tensor):
                    attn = decoder_out[1]
                else:
                    attn_holder = decoder_out[1]["attn"]
                    if isinstance(attn_holder, Tensor):
                        attn = attn_holder
                    elif attn_holder is not None:
                        attn = attn_holder[0]
                if attn is not None:
                    attn = attn[:, -1, :]

            decoder_out_tuple = (
                decoder_out[0][:, -1:, :].div(temperature),
                None if decoder_len <= 1 else decoder_out[1],
            )

            probs = model.get_normalized_probs(
                decoder_out_tuple, log_probs=True, sample=None
            )
            probs = probs[:, -1, :]
            if self.models_size == 1:
                return probs, decoder_out_emb, attn, aux_tags

            log_probs.append(probs)
            outs.append(decoder_out_emb)
            if aux_tags is not None:
                aux_tags_lprobs.append(aux_tags)
            if attn is not None:
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)
        avg_probs = torch.logsumexp(torch.stack(log_probs, dim=0), dim=0) - math.log(
            self.models_size
        )
        avg_decoder_outs = torch.sum(torch.stack(outs, dim=0), dim=0).div_(self.models_size)
        if len(aux_tags_lprobs) > 0:
            avg_aux_tags_lprobs = torch.logsumexp(torch.stack(aux_tags_lprobs, dim=0), dim=0) - math.log(
                self.models_size
            )
        if avg_attn is not None:
            avg_attn.div_(self.models_size)
        return avg_probs, avg_decoder_outs, avg_attn, avg_aux_tags_lprobs
