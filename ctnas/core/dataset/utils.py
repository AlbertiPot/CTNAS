import torch.utils.data as data

from .seq2arch import seq2arch_fn
from .tensorize import tensorize_fn


class CachedSubset(data.Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

        self.cache = [self.dataset[i] for i in self.indices]        
        """
        self.dataset[0] = (
        tensor([[0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0.],
        [1., 1., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0.],
        [1., 0., 0., 0., 1., 0., 0.],
        [0., 0., 1., 0., 0., 1., 0.]]), 
        
        tensor([1, 2, 4, 2, 2, 3, 5]), 
        0.9150974949200948,
        0.9066840211550394
        )     
        
        """

    def __getitem__(self, idx):
        return self.cache[idx]

    def __len__(self):
        return len(self.indices)


class ControllerDataset(data.Dataset):
    def __init__(self, archs):
        self.archs = archs

    def __getitem__(self, i):
        arch = self.archs[i]
        arch = tensorize_fn(seq2arch_fn(arch))
        return [*arch, 0, 0]

    def __len__(self):
        return len(self.archs)
