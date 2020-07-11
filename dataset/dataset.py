import torch
import numpy as np
from torch.utils.data import dataset, dataloader


class SteganalysisDataset(object):
    def __init__(self, path):
        self.root_path = path

    def __len__(self):
        return NotImplemented

    def __getitem__(self, item):
        return NotImplemented
