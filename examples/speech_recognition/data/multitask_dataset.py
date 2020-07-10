import torch

from fairseq.data import FairseqDataset


class MultiTaskDataset(FairseqDataset):
    def __init__(self, base_dataset, auxiliary_targets):
        super().__init__()
        self.base_dataset = base_dataset
        self.auxiliary_targets = auxiliary_targets

    def __getitem__(self, index):
        item = self.base_dataset[index]
        item['auxiliary_target'] = self.auxiliary_targets[index]
        return item

    def __len__(self):
        return len(self.base_dataset)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[int]): sample indices to collate

        Returns:
            dict: a mini-batch suitable for forwarding with a Model
        """
        batch = self.base_dataset.collater(samples)
        # In case of an empty batch, return an empty dict
        if len(batch) == 0:
            return {}
        auxiliary_targets_map = {}
        for i, s in enumerate(samples):
            auxiliary_targets_map[s['id']] = i
        sort_order = []
        for s_id in batch['id'].tolist():
            sort_order.append(auxiliary_targets_map[s_id])
        sort_order = torch.tensor(sort_order)
        auxiliary_target = torch.stack([s["auxiliary_target"] for s in samples])
        batch['auxiliary_target'] = auxiliary_target.index_select(0, sort_order)
        return batch

    def num_tokens(self, index):
        return self.base_dataset.num_tokens(index)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return self.base_dataset.size(index)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        return self.base_dataset.ordered_indices()

    @property
    def supports_prefetch(self):
        """Whether this dataset supports prefetching."""
        return (hasattr(self.base_dataset, 'supports_prefetch') and
                self.base_dataset.supports_prefetch) or \
               (hasattr(self.auxiliary_targets, 'supports_prefetch') and self.auxiliary_targets.supports_prefetch)

    def prefetch(self, indices):
        """Prefetch the data required for this epoch."""
        if self.base_dataset.supports_prefetch:
            self.base_dataset.prefetch(indices)
        if self.auxiliary_targets.supports_prefetch:
            self.auxiliary_targets.prefetch(indices)
