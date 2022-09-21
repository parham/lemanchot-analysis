
"""
    @project LeManchot-Analysis : Core components
    @organization Laval University
    @lab MiViM Lab
    @supervisor Professor Xavier Maldague
    @industrial-partner TORNGATS
"""

import glob
import os
import re

import numpy as np

from torch.utils.data import Dataset
from typing import Dict
from PIL import Image

class Huang2020Dataset(Dataset):
    """Dataset class handling the texture dataset presented by Huang et. al (2020)
        Citation: Huang, Yibin, Congying Qiu, Xiaonan Wang, Shijun Wang, and Kui Yuan.
                  "A compact convolutional neural network for surface defect inspection."
                  Sensors 20, no. 7 (2020)
        Dataset: https://github.com/abin24/Textures-Dataset
    Args:
        root_dir (str): directory path of dataset in the local storage
        categories (Dict[str,int]): the list of categories and associated label indexes
        transform (_type_): the transformation needed to be applied on the given data
        target_transform (_type_): the target transformation needed to be applied on the given target
    """

    def __init__(
        self,
        root_dir: str,
        categories: Dict[str, int],
        transform=None,
        target_transform=None,
    ) -> None:
        super().__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.categories = categories
        self.root_dir = root_dir
        self._dataset = []
        # Extract list of categories
        subdir = os.listdir(root_dir)
        for cname in subdir:
            # Extract category
            category = cname.split("_")
            if len(category) < 2:
                continue
            category = re.search(r"\D+", category[-1]).group(0)
            # Skip the category folder
            if not category in categories.keys():
                continue
            # Generate the
            files = glob.glob(os.path.join(root_dir, cname, "*.jpg"))
            for f in files:
                data = (f, category, categories[category])
                self._dataset.append(data)

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        data = self._dataset[idx]
        # Load image as a numpy array
        img = np.asarray(Image.open(data[0]))
        # Generate the target image
        target_classid = data[-1]
        target_shape = img.shape if len(img.shape) < 3 else img.shape[:-1]
        target = np.full(target_shape, target_classid)
        return (img, target)
