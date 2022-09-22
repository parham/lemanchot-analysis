
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


class MATLABDataset(Dataset):
    """
    This dataset handles loading of matlab (*.mat) files. 
    The dataset loads all mat files in a given directory.
    """
    def __init__(self,
        root_dir : str,
        input_tag : str,
        target_tag : str = None,
        transforms = None,
        target_transforms = None
    ) -> None:
        """
        Args:
            root_dir (str): The root directory containing the mat files
            input_tag (str): the label for getting the input from the loaded mat file
            target_tag (str, optional): the label for getting the target from the loaded mat file. Defaults to None.
            transforms (_type_, optional): the transformation applying to the given input. Defaults to None.
            target_transforms (_type_, optional): the transformation applying to the given target if exist. Defaults to None.

        Raises:
            ValueError: raise if the directory does not exist
        """
        super().__init__()
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.root_dir = root_dir
        # Check if the root directory exist!
        if not os.path.isdir(root_dir):
            raise ValueError('The directory "%s" does not exist.' % root_dir)
        # Extract the list of mat files.
        self.file_list = glob.glob(os.path.join(self.root_dir, '*.mat'))
        if len(self.file_list) == 0:
            raise ValueError('No mat file does not exist.')
        self.input_tag = input_tag
        self.target_tag = target_tag

    def __len__(self):
        """the count of mat files in the root directory.

        Returns:
            int: number of mat files.
        """
        return len(self.file_list)

    def __getitem__(self, idx):
        """Getting the data with the given index

        Args:
            idx (int): index of the required file

        Raises:
            ValueError: if the given input tag does not exist in the given mat file
            ValueError: if the given target tag does not exist in the given mat file

        Returns:
            Tuple: input, target, abd filename
        """
        fs = self.file_list[idx]
        # Load the mat file
        data = loadmat(fs)
        
        if not self.input_tag in data:
            raise ValueError('Input tag does not included in the data')
        if self.target_tag is not None and \
            not self.target_tag in data:
            raise ValueError('Target tag does not included in the data')
        
        input = data[self.input_tag]
        target = data[self.target_tag] if self.target_tag is not None else np.zeros(input.shape, dtype=np.uint8)

        if self.transforms is not None:
            input = self.transforms(input)
            
        if self.target_tag is not None and \
            self.target_transforms is not None:
            target = self.target_transforms(target)

        return (input, target, fs)