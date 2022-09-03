
import torch
import numpy as np

from typing import Any, List
from PIL import Image
from skimage import color

from torchvision.transforms import ToPILImage, PILToTensor
from torchvision.transforms.functional import InterpolationMode, resize

class GrayToRGB(torch.nn.Module):
    def forward(self, sample) -> Any:
        res = sample
        if len(sample.shape) < 3:
            res = np.expand_dims(res, axis=2)
            res = np.concatenate((res,res,res), axis=2)
        return res

class FilterOutAlphaChannel(torch.nn.Module):
    def forward(self, img) -> Any:
        channel = img.shape[0]
        res = img[:-1,:,:] if channel > 3 else img
        return res

class ImageResize(torch.nn.Module):
    def __init__(self, size, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=None):
        super().__init__()
        self.size = size
        self.interpolation = interpolation
        self.max_size = max_size
        self.antialias = antialias
    
    def forward(self, img):
        img_pil = Image.fromarray(np.uint8(img))
        res = resize(img_pil, self.size, self.interpolation, self.max_size, self.antialias)
        return np.asarray(res)

class ImageResizeByCoefficient(torch.nn.Module):
    def __init__(self, coefficient, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=None):
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
        res = resize(img_pil, img_size[:2], self.interpolation, self.max_size, self.antialias)
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

class ClassMapToMDTarget(torch.nn.Module):
    def __init__(self, categories : List, background_classid : int = 0) -> None:
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