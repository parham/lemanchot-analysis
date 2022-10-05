
""" 
    @title A Deep Semi-supervised Segmentation Approach for Thermographic Analysis of Industrial Components
    @organization Laval University
    @professor  Professor Xavier Maldague
    @author     Parham Nooralishahi
    @email      parham.nooralishahi@gmail.com
    @description Implementation of a W-Net CNN for unsupervised learning of image segmentations.
                 adopted from https://github.com/fkodom/wnet-unsupervised-image-segmentation
"""

import numpy as np
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from lemanchot.models.core import BaseModule, model_register

def make_same_size(src, ref):
    pt = np.array(ref.shape) - np.array(src.shape)
    pt[pt < 0] = 0
    pad = list()
    for p in pt:
        if p > 0:
            pad.insert(0, int(p // 2))
            pad.insert(0, int((p // 2) + 1))
        else:
            pad.insert(0, 0)
            pad.insert(0, 0)

    return F.pad(src, pad, mode='constant')

class ConvPoolBlock(nn.Module):
    """Performs multiple 2D convolutions, followed by a 2D max-pool operation.  Many of these are contained within
    each UNet module, for down sampling image data."""

    def __init__(self, in_features: int, out_features: int):
        """
        :param in_features: Number of feature channels in the incoming data
        :param out_features: Number of feature channels in the outgoing data
        """
        super(ConvPoolBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_features),
            nn.ReplicationPad2d(2),
            nn.Conv2d(in_features, out_features, 5),
            nn.ReplicationPad2d(1),
            nn.Conv2d(out_features, out_features, 3),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pushes a set of inputs (x) through the network.

        :param x: Input values
        :return: Module outputs
        """
        return self.layers(x)

class DeconvBlock(nn.Module):
    """
    Performs multiple 2D transposed convolutions, with a stride of 2 on the last layer.  Many of these are contained
    within each UNet module, for up sampling image data.
    """

    def __init__(self, in_features: int, out_features: int):
        """
        :param in_features: Number of feature channels in the incoming data
        :param out_features: Number of feature channels in the outgoing data
        """
        super(DeconvBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_features),
            nn.ConvTranspose2d(in_features, out_features, 5, padding=2),
            nn.ConvTranspose2d(out_features, out_features, 2, stride=2),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pushes a set of inputs (x) through the network.

        :param x: Input values
        :return: Module outputs
        """
        return self.layers(x)

class OutputBlock(nn.Module):
    """
    Performs multiple 2D convolutions, without any pooling or strided operations.
    """

    def __init__(self, in_features: int, out_features: int):
        """
        :param in_features: Number of feature channels in the incoming data
        :param out_features: Number of feature channels in the outgoing data
        """
        super(OutputBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(in_features, out_features, 3),
            nn.ReplicationPad2d(1),
            nn.Conv2d(out_features, out_features, 3),
            nn.Conv2d(out_features, out_features, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pushes a set of inputs (x) through the network.

        :param x: Input values
        :return: Module outputs
        """
        return self.layers(x)

class UNetEncoder(nn.Module):
    """
    The first half (encoder) of the W-Net architecture.  
    Returns class probabilities for each pixel in the image.
    """

    def __init__(self, num_channels: int = 3, num_classes: int = 10):
        """
        :param num_channels: Number of channels in the raw image data
        :param num_classes: Number of classes in the output class probabilities
        """
        super(UNetEncoder, self).__init__()
        self.conv1 = ConvPoolBlock(num_channels, 32)
        self.conv2 = ConvPoolBlock(32, 32)
        self.conv3 = ConvPoolBlock(32, 64)
        self.deconv1 = DeconvBlock(64, 32)
        self.deconv2 = DeconvBlock(64, 32)
        self.deconv3 = DeconvBlock(64, 32)
        self.output = OutputBlock(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pushes a set of inputs (x) through the network.

        :param x: Input values
        :return: Network outputs
        """

        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        x = self.conv3(c2)
        x = self.deconv1(x)
        x = make_same_size(x, c2)
        x = self.deconv2(torch.cat((c2, x), dim=1))
        x = make_same_size(x, c1)
        x = self.deconv3(torch.cat((c1, x), dim=1))
        x = self.output(x)
        return x

class UNetDecoder(nn.Module):
    """
    The second half (decoder) of the W-Net architecture.  
    Returns a reconstruction of the original image.
    """

    def __init__(self, num_channels: int = 3, num_classes: int = 20):
        """
        :param num_channels: Number of channels in the raw image data
        :param num_classes: Number of classes in the output class probabilities
        """
        super(UNetDecoder, self).__init__()
        self.conv1 = ConvPoolBlock(num_classes, 32)
        self.conv2 = ConvPoolBlock(32, 32)
        self.conv3 = ConvPoolBlock(32, 64)
        self.deconv1 = DeconvBlock(64, 32)
        self.deconv2 = DeconvBlock(64, 32)
        self.deconv3 = DeconvBlock(64, 32)
        self.output = OutputBlock(32, num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pushes a set of inputs (x) through the network.

        :param x: Input values
        :return: Network outputs
        """
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        x = self.conv3(c2)
        x = self.deconv1(x)
        x = make_same_size(x, c2)
        x = self.deconv2(torch.cat((c2, x), dim=1))
        x = make_same_size(x, c1)
        x = self.deconv3(torch.cat((c1, x), dim=1))
        x = self.output(x)

        return x

@model_register('wnet')
class WNet(BaseModule):
    """
    Implements a W-Net CNN model for learning unsupervised image segmentations.  
    First encodes image data into class probabilities using UNet, and then decodes 
    the labels into a reconstruction of the original image using a second UNet.
    """

    def __init__(self, name : str, config) -> None:
        """
        :param num_channels: Number of channels in the raw image data
        :param num_classes: Number of classes in the output class probabilities
        """
        super().__init__(
            name='wnet',
            config=config
        )
        self.encoder = UNetEncoder(
            num_channels=self.num_channels, 
            num_classes=self.num_classes)
        self.decoder = UNetDecoder(
            num_channels=self.num_channels, 
            num_classes=self.num_classes)

    def forward_encode_(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pushes a set of inputs (x) through only the encoder network.

        :param x: Input values
        :return: Class probabilities
        """

        return self.encoder(x)

    def forward_reconstruct_(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Pushes a set of class probabilities (mask) through only the decoder network.

        :param mask: Class probabilities
        :return: Reconstructed image
        """

        outputs = self.decoder(mask)
        outputs = nn.ReLU()(outputs)

        return outputs

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pushes a set of inputs (x) through the network.

        :param x: Input values
        :return: Network outputs
        """

        encoded = self.forward_encode_(x).transpose(1, -1)
        mask = nn.Softmax(-1)(encoded).transpose(-1, 1)
        reconstructed = self.forward_reconstruct_(mask)

        # return mask, reconstructed
        return mask, reconstructed
