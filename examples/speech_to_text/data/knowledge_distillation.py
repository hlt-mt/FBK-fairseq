# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import numpy as np
import torch
from torch.nn import functional as F

from fairseq.data import IndexedCachedDataset, FairseqDataset
from fairseq.data.indexed_dataset import IndexedDatasetBuilder


class TeacherOutputDatasetBuilder(IndexedDatasetBuilder):
    def add_item(self, data):
        # +1 for Lua compatibility
        data = np.array(data, dtype=self.dtype)
        bytes = self.out_file.write(data)
        self.data_offsets.append(self.data_offsets[-1] + bytes / self.element_size)
        for s in data.shape:
            self.sizes.append(s)
        self.dim_offsets.append(self.dim_offsets[-1] + len(data.shape))


class TeacherOutputDataset(IndexedCachedDataset):

    def __init__(self, prefix, dtype):
        self.cache_index = {}
        super().__init__(prefix, fix_lua_indexing=False)
        self.dtype=dtype

    @staticmethod
    def save_bin(prefix, data_list, dtype=np.float):
        bin_path = prefix + '.bin'
        idx_path = prefix + '.idx'
        builder = TeacherOutputDatasetBuilder(bin_path, dtype)
        for d in data_list:
            builder.add_item(d)
        builder.finalize(idx_path)

    def __getitem__(self, i):
        self.check_index(i)
        tensor_size = self.sizes[self.dim_offsets[i]:self.dim_offsets[i + 1]]
        a = np.empty(tensor_size, dtype=self.dtype)
        ptx = self.cache_index[i]
        np.copyto(a, self.cache[ptx: ptx + a.size].reshape(tensor_size))
        item = torch.from_numpy(a)
        if self.dtype in [np.int32, np.int64, np.int16, np.int8, np.uint8]:
            item = item.long()
        else:
            item = item.float()
        return item


class DatasetWithTeacherOutput(FairseqDataset):
    """
    A wrapper dataset for fairseq.data.FairseqDataset which adds the teacher outputs
    to it.
    Args:
        src (fairseq.data.FairseqDataset): source dataset to wrap
        teacher_idxs (torch.utils.data.Dataset): dataset of indexes of teacher output probabilities
        teacher_probs (torch.utils.data.Dataset): dataset of probabilities of teacher output
        tgt_dict (~fairseq.data.Dictionary): target vocabulary
        distill_k (int): number of indexes to learn from teacher distribution
    """
    def __init__(self, src, teacher_probs, teacher_idxs, tgt_dict, distill_k):
        self.src = src
        self.teacher_probs = teacher_probs
        self.teacher_idxs = teacher_idxs
        self.tgt_dict = tgt_dict
        self.distill_k = distill_k

    def __getitem__(self, index):
        item = self.src.__getitem__(index)
        teacher_idxs, teacher_probs = self.teacher_idxs[index], self.teacher_probs[index]
        item['teacher_output'] = [teacher_idxs, teacher_probs]
        return item

    def __len__(self):
        return len(self.src)

    def num_tokens(self, index):
        return self.src.num_tokens(index)

    def size(self, index):
        return self.src.size(index)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch with the following keys:
                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:
                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one position
                    for input feeding/teacher forcing, of shape `(bsz,
                    tgt_len)`. This key will not be present if *input_feeding*
                    is ``False``. Padding will appear on the left if
                    *left_pad_target* is ``True``.
                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
                - `teacher_output` (Tuple[LongTensor, FloatTensor]): a tuple which
                  contains two tensors both of sizes `(bsz, tgt_len, distill_topk)`
                  where `distill_topk` is the number of items considered of the
                  teacher output.
        """
        batch = self.src.collater(samples)
        if len(samples) > 0:
            tgt_len = batch['target'].shape[1]
            pad_idx = self.tgt_dict.pad()
            teacher_outputs = {}
            for s in samples:
                teacher_outputs[s['id']] = [
                    F.pad(s['teacher_output'][0], (0, 0, 0, tgt_len - s['teacher_output'][0].shape[0]), value=pad_idx),
                    F.pad(s['teacher_output'][1], (0, 0, 0, tgt_len - s['teacher_output'][1].shape[0]))
                ]
            teacher_idxs = []
            teacher_probs = []
            for s_id in batch['id'].tolist():
                teacher_idxs.append(teacher_outputs[s_id][0])
                teacher_probs.append(teacher_outputs[s_id][1])
            batch['teacher_output'] = [torch.stack(teacher_idxs), torch.stack(teacher_probs)]
        return batch

    def prefetch(self, indices):
        self.src.prefetch(indices)
        self.teacher_probs.prefetch(indices)
        self.teacher_idxs.prefetch(indices)

    @property
    def supports_prefetch(self):
        return self.src.supports_prefetch and (
            hasattr(self.teacher_probs, 'supports_prefetch')
            and self.teacher_probs.supports_prefetch
            and hasattr(self.teacher_idxs, 'supports_prefetch')
            and self.teacher_idxs.supports_prefetch)