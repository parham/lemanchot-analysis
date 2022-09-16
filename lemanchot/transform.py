"""
    @project LeManchot-Analysis : Core components
    @organization Laval University
    @lab MiViM Lab
    @supervisor Professor Xavier Maldague
    @industrial-partner TORNGATS
"""

import torch
import numpy as np
from random import choices
from typing import Any, List, Tuple
from PIL import Image
from skimage import color

from torchvision.transforms import ToPILImage, PILToTensor, RandomCrop
from torchvision.transforms.functional import InterpolationMode, resize, rotate, crop


class GrayToRGB(torch.nn.Module):
    def forward(self, sample) -> Any:
        res = sample
        if len(sample.shape) < 3:
            res = np.expand_dims(res, axis=2)
            res = np.concatenate((res, res, res), axis=2)
        return res

class FilterOutAlphaChannel(torch.nn.Module):
    def forward(self, img) -> Any:
        channel = img.shape[0]
        res = img[:-1, :, :] if channel > 3 else img
        return res

class BothRandomRotate(torch.nn.Module):
    def __init__(self, angles: Tuple[int], weights: Tuple[int] = None):
        super().__init__()
        self.angles = angles
        self.weights = weights if not weights else [1] * len(angles)

    def forward(self, args):
        ang = choices(self.angles, weights=self.weights, k=1)[0]
        return [rotate(img, ang) for img in args]

class BothRandomCrop(torch.nn.Module):
    def __init__(self, crop_size):
        super().__init__()
        self.size = crop_size

    def forward(self, args):
        i, j, h, w = RandomCrop.get_params(args[0], self.size)
        return [crop(img, i, j, h, w) for img in args]

    def forward(self, args):
        i, j, h, w = RandomCrop.get_params(args[0], self.size)
        return [crop(img, i, j, h, w) for img in args]

class ImageResize(torch.nn.Module):
    def __init__(
        self,
        size,
        interpolation=InterpolationMode.BILINEAR,
        max_size=None,
        antialias=None,
    ):
        super().__init__()
        self.size = size
        self.interpolation = interpolation
        self.max_size = max_size
        self.antialias = antialias

    def forward(self, img):
        img_pil = Image.fromarray(np.uint8(img))
        res = resize(
            img_pil, self.size, self.interpolation, self.max_size, self.antialias
        )
        return np.asarray(res)

class ImageResizeByCoefficient(torch.nn.Module):
    def __init__(
        self,
        coefficient,
        interpolation=InterpolationMode.BILINEAR,
        max_size=None,
        antialias=None,
    ):
        super().__init__()
        self.coefficient = coefficient
        self.interpolation = interpolation
        self.max_size = max_size
        self.antialias = antialias

    def forward(self, img):
        img_size = list(img.shape)
        img_size[0] = (img_size[0] // self.coefficient) * self.coefficient
        img_size[1] = (img_size[1] // self.coefficient) * self.coefficient

        img_pil = Image.fromarray(np.uint8(img))
        res = resize(
            img_pil, img_size[:2], self.interpolation, self.max_size, self.antialias
        )
        return np.asarray(res)

class NumpyImageToTensor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.to_pil = ToPILImage()
        self.to_tensor = PILToTensor()

    def forward(self, img):
        img = self.to_pil(img)
        img = self.to_tensor(img)
        return img

class ToGrayscale(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, img):
        return color.rgb2gray(img) if len(img.shape) > 2 else img


class TargetDilation(torch.nn.Module):
    def __init__(self, factor) -> None:
        super().__init__()
        self.kernel = torch.ones((1, 1, factor, factor), requires_grad=False, dtype=torch.uint8)

    def forward(self, img: Image):
        return torch.clamp(
            torch.nn.functional.conv2d(img, self.kernel.to(img.dtype), padding="same"), 0, 1
        )


class ClassMapToMDTarget(torch.nn.Module):
    def __init__(self, categories: List, background_classid: int = 0) -> None:
        super().__init__()
        self.categories = categories
        self.background_classid = background_classid

    def forward(self, img):
        print(img)
        gt = np.zeros(img.shape)
        tt = np.ones(img.shape)
        # layers = ((np.ones(img.shape) * self.background_classid), *[np.where(img == i, img, gt) for i in self.categories])
        layers = []
        for i in self.categories:
            tmp = np.where(img == i, tt, gt)
            layers.append(tmp)
        layers = ((np.ones(img.shape) * self.background_classid), *layers)
        return np.stack(layers)