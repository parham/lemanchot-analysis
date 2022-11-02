"""
    @project LeManchot-Analysis : Core components
    @organization Laval University
    @lab MiViM Lab
    @supervisor Professor Xavier Maldague
    @industrial-partner TORNGATS
"""

import torch
from torch import Tensor, logical_not
import numpy as np
from random import choices
from typing import Any, List, Tuple, Optional
from PIL import Image
from skimage import color

from torchvision.transforms import RandomCrop
from torchvision.transforms.autoaugment import TrivialAugmentWide as TAWide, _apply_op
from torchvision.transforms.functional import (
    InterpolationMode,
    resize,
    rotate,
    crop,
    get_dimensions,
    to_tensor,
    normalize,
)

from lemanchot.core import get_device


class GrayToRGB(torch.nn.Module):
    def forward(self, sample) -> Any:
        res = sample
        if len(sample.shape) < 3:
            res = np.expand_dims(res, axis=2)
            res = np.concatenate((res, res, res), axis=2)
        return res


class FilterOutAlphaChannel(torch.nn.Module):
    def forward(self, img) -> Any:
        return img[:3, :, :] if len(img.shape) == 3 and img.shape[0] > 3 else img


class BothRandomRotate(torch.nn.Module):
    def __init__(self, angles: Tuple[int], weights: Tuple[int] = None):
        super().__init__()
        self.angles = angles
        self.weights = weights if not weights else [1] * len(angles)

    def forward(self, img, target):
        ang = choices(self.angles, weights=self.weights, k=1)[0]
        return rotate(img, ang), rotate(target, ang)


class BothRandomCrop(torch.nn.Module):
    def __init__(self, crop_size):
        super().__init__()
        self.size = crop_size

    def forward(self, img, target):
        i, j, h, w = RandomCrop.get_params(img, self.size)
        return crop(img, i, j, h, w), crop(target, i, j, h, w)


class BothToTensor(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img, target):
        return to_tensor(img), target


class BothNormalize(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, img, target):
        return normalize(img, self.mean, self.std), target


class BothCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


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

    def forward(self, img):
        tmp = torch.tensor(img, device=get_device())
        if len(tmp.shape) == 3:
            tmp = tmp.permute((-1, 0, 1))

        return tmp


class ToFloatTensor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, img):
        return img.to(dtype=torch.float)


class ToLongTensor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, img):
        return img.to(dtype=torch.long)


class ToUINT8Tensor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, img):
        return img.to(dtype=torch.uint8)


class ToGrayscale(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, img):
        return color.rgb2gray(img) if len(img.shape) > 2 else img


class TargetDilation(torch.nn.Module):
    def __init__(self, factor, channel: int = 1) -> None:
        super().__init__()
        self.kernel = torch.ones(
            (1, 1, factor, factor), requires_grad=False, dtype=torch.uint8
        )
        self.channel = channel

    def forward(self, img: Image):
        if img.size(0) == 2:
            img[1, ...] = torch.clamp(
                torch.nn.functional.conv2d(
                    img[1:, ...], self.kernel.to(img.dtype), padding="same"
                ),
                0,
                1,
            )
            img[0, ...] = logical_not(img[1:, ...])
            return img
        elif img.size(0) == 1:
            return torch.clamp(
                torch.nn.functional.conv2d(
                    img, self.kernel.to(img.dtype), padding="same"
                ),
                0,
                1,
            )
        else:
            raise NotImplementedError(
                "Dilation not implemented for targets with more than 2 channels."
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


class TrivialAugmentWide(TAWide):
    """Dataset-independent data-augmentation with TrivialAugment Wide, as described in
    `"TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation" <https://arxiv.org/abs/2103.10158>`_.
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        num_magnitude_bins (int): The number of different magnitude values.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
    """

    def __init__(
        self,
        num_magnitude_bins: int = 31,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        fill: Optional[List[float]] = None,
    ) -> None:
        super().__init__(num_magnitude_bins, interpolation, fill)

    def forward(self, img: Tensor, target: Tensor) -> Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        fill = self.fill
        channels, height, width = get_dimensions(img)
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            elif fill is not None:
                fill = [float(f) for f in fill]

        op_meta = self._augmentation_space(self.num_magnitude_bins)
        op_index = int(torch.randint(len(op_meta), (1,)).item())
        op_name = list(op_meta.keys())[op_index]
        magnitudes, signed = op_meta[op_name]
        magnitude = (
            float(
                magnitudes[
                    torch.randint(len(magnitudes), (1,), dtype=torch.long)
                ].item()
            )
            if magnitudes.ndim > 0
            else 0.0
        )
        if signed and torch.randint(2, (1,)):
            magnitude *= -1.0

        img = _apply_op(
            img, op_name, magnitude, interpolation=self.interpolation, fill=fill
        )
        if op_name in {
            "ShearX",
            "ShearY",
            "TranslateX",
            "TranslateY",
            "Rotate",
            "Invert",
        }:
            target = _apply_op(
                target, op_name, magnitude, interpolation=self.interpolation, fill=fill
            )

        return img, target
