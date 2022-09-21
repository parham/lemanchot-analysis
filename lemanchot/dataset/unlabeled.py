
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
from pathlib import Path
from typing import Callable, List, Optional, Union
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset


class UnlabeledImageDataset(Dataset):
    """
    Dataset for images in a folder
    """
    def __init__(self,
        root_dir : str,
        file_extension : str,
        transform=None,
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
        self.root_dir = root_dir
        if os.path.isdir(root_dir):
            raise ValueError('The directory "%s" does not exist.' % root_dir)
        self.file_list = glob.glob(os.path.join(self.root_dir, f'*.{file_extension}'))
        if len(self.file_list) == 0:
            raise ValueError('No %s file does not exist.' % file_extension)

    def __len__(self):
        return len(self.file_list)
    
    @functools.lru_cache(maxsize=10)
    def __getitem__(self, idx):
        fs = self.file_list[idx]
        img = np.asarray(Image.open(fs))
        if self.transforms is not None:
            img = self.transforms(img)
        return (img, None)


class ImageDataset(VisionDataset):
    def __init__(
        self,
        root: Optional[Union[str, None]] = None,
        folder_name: Optional[Union[str, None]] = None,
        paths: Optional[Union[List, None]] = None,
        transforms: Optional[Callable] = None,
    ):
        """
        Dataset Class to handle image files.
        Args:
            root (str): Optional. Root folder path of the dataset.
            folder_name (str): Optional. Directory name of the folder containint the image files.
            paths: List: Optional. List of input paths.
            transforms (Optional[Callable], optional):
                Transformations to be applied once the images are loaded. Defaults to None.
        Raises:
            RuntimeError: If conflicting inputs are passed. (root, folder_name) vs paths.
            ValueError: If the paths are not found.
        """
        super().__init__(root, transforms)

        if (root and folder_name) and paths:
            raise RuntimeError(
                "Only one of (root, folder_name) or paths must be passed, not both."
            )
        if (root and folder_name) and not os.path.isdir(
            os.path.join(root, folder_name)
        ):
            raise ValueError(
                "The dataset directory does not exist or is not valid!",
                os.path.join(root, folder_name),
            )

        # Get the list of files in the dataset directory
        if paths is None:
            self.paths = [str(p) for p in list(Path(root).rglob(f"{folder_name}/*"))]
        else:
            self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):

        path = self.paths[index]
        img = Image.open(path).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)

        return path, img