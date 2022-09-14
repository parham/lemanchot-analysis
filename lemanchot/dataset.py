
""" 
    @project LeManchot-Analysis : Core components
    @organization Laval University
    @lab MiViM Lab
    @supervisor Professor Xavier Maldague
    @industrial-partner TORNGATS
"""

import functools
import glob
import os
import numpy as np

from PIL import Image
from typing import Dict
from torch.utils.data import Dataset
from gimp_labeling_converter import XCFDataset, generate_cmap, gimp_helper

class FileRepeaterDataset(Dataset):

    __xcf_filext = '.xcf'

    def __init__(self, file : str, 
        category : Dict, iteration : int = 1,
        transform = None,
        target_transform = None
    ) -> None:
        super().__init__()

        self.iteration = iteration
        if not os.path.isfile(file) or \
            not file.endswith(self.__xcf_filext):
            raise ValueError(f'{file} is invalid!')

        res = generate_cmap(file=file, helper=gimp_helper, category=category)
        self.image = np.asarray(res['original'])
        self.target = np.asarray(res['target'])
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

class RepetitiveDatasetWrapper(XCFDataset):
    def __init__(self, 
        root_dir: str, 
        category: Dict[str, int], 
        transform=None, 
        target_transform=None
    ) -> None:
        super().__init__(root_dir, category, transform, target_transform)

    @property
    def wrapped_dataset(self):
        return self.dataset_

    def __len__(self):
        return super().__len__() * self.iteration

    def __getitem__(self, idx):
        super().__getitem__(idx % self.actual_size)


class ImageDataset(Dataset):
    """
    Dataset for images in a folder
    """
    def __init__(self,
        root_dir : str,
        file_extension : str,
        transform=None, 
        target_transform=None
    ) -> None:
        """
        Args:
            root_dir (str): root directory containing images
            file_extension (str): the targeted file extension
            transform (_type_, optional): the transformations for the images. Defaults to None.
            target_transform (_type_, optional): the transformations for the targets. Defaults to None.
        """ 
        super().__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.root_dir = root_dir
        self.file_list = glob.glob(os.path.join(self.root_dir, f'*.{file_extension}'))

    def __len__(self):
        return len(self.file_list)
    
    @functools.lru_cache(maxsize=10)
    def __getitem__(self, idx):
        fs = self.file_list[idx]
        img = np.asarray(Image.open(fs))
        return (img, None)
