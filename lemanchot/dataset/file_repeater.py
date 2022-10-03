
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
    """
    This dataset repeats a file for the number of time determined in the configuration file
    """

    __xcf_filext = ".xcf"

    def __init__(
        self,
        file: str,
        category: Dict,
        iteration: int = 1,
        transform=None,
        target_transform=None,
    ) -> None:
        """
        Args:
            file (str): The file path associated to the unlabeled data
            category (Dict): the list of categories and associated id
            iteration (int, optional): the number of times that the data need to be repeated. Defaults to 1.
            transform (_type_, optional): the transformation for the given input. Defaults to None.
            target_transform (_type_, optional): the transformation for the given target. Defaults to None.

        Raises:
            ValueError: raise if file does not exist or the file is not with the correct file extension.
        """
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

    def __len__(self) -> int:
        """The number of iterations

        Returns:
            int: number of iteration
        """
        return self.iteration

    def __getitem__(self, idx):
        """Get the item associated to the index. This dataset returns the same given data for all indexes.

        Args:
            idx (int): not used!

        Returns:
            Tuple: a tuple containing the given input and target data
        """
        img = self.image
        target = self.target
        # Apply the transformation to the input before return
        if self.transform is not None:
            img = self.transform(self.image)
        # Apply the transformation to the target before return
        if self.target_transform is not None:
            target = self.target_transform(self.target)
        return img, target