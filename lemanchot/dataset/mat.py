
"""
    @project LeManchot-Analysis : Core components
    @organization Laval University
    @lab MiViM Lab
    @supervisor Professor Xavier Maldague
    @industrial-partner TORNGATS
"""

import os
import glob
import numpy as np

from scipy.io import savemat, loadmat
from torch.utils.data import Dataset


class UnlabeledImageDataset(Dataset):
    def __init__(self,
        root_dir : str,
        input_tag : str,
        target_tag : str,
        transforms=None,
        target_transforms=None,
    ) -> None:
        super().__init__()
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.root_dir = root_dir
        self.file_list = glob.glob(os.path.join(self.root_dir, '*.mat'))
        self.input_tag = input_tag
        self.target_tag = target_tag

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fs = self.file_list[idx]
        # Load the mat file
        data = loadmat(fs)
        
        if not self.input_tag in data:
            raise ValueError('Input tag does not included in the data')
        if not self.target_tag in data:
            raise ValueError('Target tag does not included in the data')
        
        input = data[self.input_tag]
        target = data[self.target_tag]

        if self.transforms is not None:
            input = self.transforms(input)
        if self.target_transforms is not None:
            target = self.target_transforms(target)

        return (input, target)