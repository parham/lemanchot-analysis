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
import json
import numpy as np
from pathlib import Path

from PIL import Image
from typing import Dict, Optional, Set, List, Callable, Tuple, Union
from torch import Tensor, from_numpy
from torch import stack as torch_stack
from torch import zeros as torch_zeros
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
from gimp_labeling_converter import XCFDataset, generate_cmap, gimp_helper
from .rle import decode_rle


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

        self.iteration = iteration
        if not os.path.isfile(file) or not file.endswith(self.__xcf_filext):
            raise ValueError(f"{file} is invalid!")

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


class RepetitiveDatasetWrapper(XCFDataset):
    def __init__(
        self,
        root_dir: str,
        category: Dict[str, int],
        transform=None,
        target_transform=None,
    ) -> None:
        super().__init__(root_dir, category, transform, target_transform)

    @property
    def wrapped_dataset(self):
        return self.dataset_

    def __len__(self):
        return super().__len__() * self.iteration

    def __getitem__(self, idx):
        super().__getitem__(idx % self.actual_size)


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


class JSONDataset(VisionDataset):
    """
    Dataset Class to handle RLEs encoded masks in JSON files.
    """

    def __init__(
        self,
        root: str,
        folder_name: str,
        classes: Set,
        transforms: Optional[Callable] = None,
    ):
        """
        Dataset Class to handle RLEs encoded masks in JSON files.
        Args:
            root (str): Root folder path of the dataset.
            folder_name (str): Directory name of the folder containint the JSON files.
            transforms (Optional[Callable], optional):
                Transformations to be applied once the targets are decoded. Defaults to None.
        Raises:
            ValueError: If the paths are not found.
        """
        super().__init__(root, transforms)
        self.folder_name = folder_name
        self.classes = classes

        if not os.path.isdir(self.root):
            raise ValueError("The dataset directory does not exist or is not valid!")

        self.paths = [str(p) for p in list(Path(root).rglob(f"{folder_name}/*.json"))]

    def __len__(self) -> int:
        return len(self.paths)

    def JSON2ClassMap(self, input: Dict) -> Dict:
        """
        Decoded and encoded RLE generated by the `generateJSON` function.

        Notes:
            Class indexes in the returned Array are in order of appearance in the JSON file.
            Annotations of the same class are added to the same index.

        Args:
            input (Dict): Decoded JSON file in a dictionary format.
            filter (Callable): Function used to filter and order decoded classes.
                Must take class set as input and return filtered class set as output.

        Returns:
            Dict: Decoded tensor in one-hot format class: tensor(W, H)
        """
        height = input["height"]
        width = input["width"]
        layers = {}
        for cl, ann in input["annotations"].items():
            if ann.get("data", False):
                layers[cl] = from_numpy(
                    decode_rle(ann["data"]).reshape(height, width, 4)[:, :, 3]
                )

        return layers

    def __getitem__(self, index: int) -> Tuple[str, Tensor]:
        """
        Return a path
        Args:
            index (int): Index of the file to be decoded.
        Returns:
            Tuple[str, Tensor]: Path of the decoded JSON and the decoded mask.
        """
        path = self.paths[index]
        with open(path, "r") as f:
            data = json.load(f)

        size = (data["height"], data["width"])
        target = self.JSON2ClassMap(data)
        target = torch_stack(
            [target.get(c, torch_zeros(size)) for c in self.classes], dim=0
        )

        if self.transforms is not None:
            target = self.transforms(target)

        return path, target


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
        img = Image.open(path)
        if self.transforms is not None:
            img = self.transforms(img)

        return path, img


class SegmentationDataset(VisionDataset):
    def __init__(
        self,
        root: str,
        classes: List,
        img_folder: str = "img",
        img_ext: str = ".jpg",
        gt_folder: str = "gt",
        input_transforms: Optional[Callable] = None,
        target_transforms: Optional[Callable] = None,
        both_transforms: Optional[Callable] = None,
    ):
        """
        Dataset Class to handle image files.
        Args:
            root (str): Root folder path of the dataset.
                Samples are placed in a folder in root/imgs and targets are in root/gts.
                Targets take priority when loading a sample, meaning the path of the
                samples is generated by replacing `gts` with `imgs`. Concequently, targets
                and samples share names and image format.
            gt_folder (str): Name of the ground truth folder.
            *_transform (Callable): Collection of transformations to be applied to the inputs
                targets, separetly of together.
        Raises:
            ValueError: If the paths are not found.
        """
        super().__init__(root)

        self.gt_dataset = JSONDataset(root, gt_folder, classes, target_transforms)
        img_paths = [
            p.replace(gt_folder, img_folder).replace(".json", img_ext)
            for p in self.gt_dataset.paths
        ]
        self.samples_dataset = ImageDataset(
            paths=img_paths, transforms=input_transforms
        )
        self.both_transforms = both_transforms

    def __len__(self) -> int:
        return len(self.gt_dataset)

    def __getitem__(self, index: int):

        path, sample = self.samples_dataset[index]
        _, target = self.gt_dataset[index]

        if self.both_transforms is not None:
            sample, target = self.both_transforms(input=[sample, target])

        return sample, target
