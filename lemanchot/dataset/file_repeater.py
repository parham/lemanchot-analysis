
"""
    @project LeManchot-Analysis : Core components
    @organization Laval University
    @lab MiViM Lab
    @supervisor Professor Xavier Maldague
    @industrial-partner TORNGATS
"""

import os
import numpy as np
from typing import Dict

from torch.utils.data import Dataset

from gimp_labeling_converter import generate_cmap, gimp_helper

class FileRepeaterDataset(Dataset):

    __xcf_filext = ".xcf"

    def __init__(
        self,
        file: str,
        category: Dict,
        iteration: int = 1,
        transform=None,
        target_transform=None,
    ) -> None:
        super().__init__()
        # Set the iteration
        self.iteration = iteration
        # Check if the given file exists
        if not os.path.isfile(file) or \
           not file.endswith(self.__xcf_filext):
            raise ValueError(f"{file} is invalid!")
        # Generate the class map from the file
        res = generate_cmap(file=file, helper=gimp_helper, category=category)
        self.image = np.asarray(res["original"])
        self.target = np.asarray(res["target"])
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.iteration

    def __getitem__(self, idx):
        img = self.image
        target = self.target
        if self.transform is not None:
            img = self.transform(self.image)
        if self.target_transform is not None:
            target = self.target_transform(self.target)
        return img, target