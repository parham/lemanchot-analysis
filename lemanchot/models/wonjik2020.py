
""" 
    @title A Deep Semi-supervised Segmentation Approach for Thermographic Analysis of Industrial Components
    @organization Laval University
    @professor  Professor Xavier Maldague
    @author     Parham Nooralishahi
    @email      parham.nooralishahi@gmail.com
"""

from dotmap import DotMap

import torch.nn as nn
import torch.nn.functional as F

from lemanchot.models.core import model_register

@model_register('wonjik2020')
class Wonjik2020Module (nn.Module):
    """ 
    Implementation of the model presented in:
    @name           Unsupervised Learning of Image Segmentation Based on Differentiable Feature Clustering   
    @journal        IEEE Transactions on Image Processing
    @year           2020
    @repo           https://github.com/kanezaki/pytorch-unsupervised-segmentation-tip
    @citation       Wonjik Kim*, Asako Kanezaki*, and Masayuki Tanaka. Unsupervised Learning of Image Segmentation Based on Differentiable Feature Clustering. IEEE Transactions on Image Processing, accepted, 2020.
    """

    def __init__(self, name : str, config) -> None:
        super().__init__(
            name='wonjik2020',
            config=config
        )

        # First convolutional layer
        self.conv1 = nn.Conv2d(
            self.num_dim, 
            self.num_channel,
            kernel_size=3, 
            stride=1, 
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(self.num_channel)
        # Feature space including multiple convolutional layers
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        # The number of convolutional layers are determined based on "nCov" parameters.
        for i in range(self.num_convs-1):
            tmpConv = nn.Conv2d(
                self.num_channel, 
                self.num_channel,
                kernel_size=3, 
                stride=1, 
                padding=1
            )
            tmpBatch = nn.BatchNorm2d(self.num_channel)
            self.conv2.append(tmpConv)
            self.bn2.append(tmpBatch)
        # The reference normalization for extracting class labels
        self.conv3 = nn.Conv2d(
            self.num_channel, 
            self.num_channel, 
            kernel_size=1, 
            stride=1, 
            padding=0
        )
        self.bn3 = nn.BatchNorm2d(self.num_channel)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        for i in range(self.num_convs-1):
            x = self.conv2[i](x)
            x = F.relu(x)
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x
