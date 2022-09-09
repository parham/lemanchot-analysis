
""" 
    @title A Deep Semi-supervised Segmentation Approach for Thermographic Analysis of Industrial Components
    @organization Laval University
    @professor  Professor Xavier Maldague
    @author     Parham Nooralishahi
    @email      parham.nooralishahi@gmail.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any, Dict
from scipy.ndimage import grey_opening

from lemanchot.filter import gaussian_kernel
from lemanchot.loss.core import BaseLoss, loss_register

@loss_register('wnet_loss')
class WNetLoss(BaseLoss):
    def __init__(self, name : str, config) -> None:
        """
        Args:
            device (str): _description_
            config (Dict[str,Any]): _description_
        Defaults:
            alpha = 1e-3, 
            beta = 1, 
            gamma = 1e-1
        """
        super().__init__(name=name, config=config)

    def forward(self, output, target, input, mask): # labels > target
        input, label, output = input.contiguous(), target.contiguous(), output.contiguous()
        # Weights for NCutLoss2D, MSELoss, and OpeningLoss2D, respectively
        ncut_loss = self.alpha * NCutLoss2D()(mask, input)
        mse_loss = self.beta * nn.MSELoss()(output, input.detach())
        smooth_loss = self.gamma * OpeningLoss2D()(mask)
        loss = ncut_loss + mse_loss + smooth_loss
        return loss

@loss_register('ncut2d_loss')
class NCutLoss2D(BaseLoss):
    """
        Implementation of the continuous N-Cut loss, as in:
        'W-Net: A Deep Model for Fully Unsupervised Image Segmentation', by Xia, Kulis (2017)
        adopted from https://github.com/fkodom/wnet-unsupervised-image-segmentation
    """
    def __init__(self, name : str, config) -> None:
        """
        :param radius: Radius of the spatial interaction term
        :param sigma_1: Standard deviation of the spatial Gaussian interaction
        :param sigma_2: Standard deviation of the pixel value Gaussian interaction
        """
        super().__init__(name=name, config=config)

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor, **kwargs) -> torch.Tensor:
        """
            Computes the continuous N-Cut loss, given a set of class probabilities (labels) and raw images (inputs).
            Small modifications have been made here for efficiency -- specifically, we compute the pixel-wise weights
            relative to the class-wide average, rather than for every individual pixel.

            :param labels: Predicted class probabilities
            :param inputs: Raw images
            :return: Continuous N-Cut loss
        """
        num_classes = labels.shape[1]
        kernel = gaussian_kernel(
            radius=self.radius, sigma=self.sigma_1, device=labels.device.type)
        loss = 0

        for k in range(num_classes):
            # Compute the average pixel value for this class, and the difference from each pixel
            class_probs = labels[:, k].unsqueeze(1)
            class_mean = torch.mean(inputs * class_probs, dim=(2, 3), keepdim=True) / \
                torch.add(torch.mean(
                    class_probs, dim=(2, 3), keepdim=True), 1e-5)
            diff = (inputs - class_mean).pow(2).sum(dim=1).unsqueeze(1)

            # Weight the loss by the difference from the class average.
            weights = torch.exp(diff.pow(2).mul(-1 / self.sigma_2 ** 2))

            # Compute N-cut loss, using the computed weights matrix, and a Gaussian spatial filter
            numerator = torch.sum(
                class_probs * F.conv2d(class_probs * weights, kernel, padding=self.radius))
            denominator = torch.sum(
                class_probs * F.conv2d(weights, kernel, padding=self.radius))
            loss += nn.L1Loss()(numerator / torch.add(denominator, 1e-6),
                                torch.zeros_like(numerator))

        return num_classes - loss

@loss_register('opening2d_loss')
class OpeningLoss2D(BaseLoss):
    """
        Computes the Mean Squared Error between computed class probabilities their grey opening.  Grey opening is a
        morphology operation, which performs an erosion followed by dilation.  Conceptually, this encourages the network
        to return sharper boundaries to objects in the class probabilities.

        NOTE:  Original loss term -- not derived from the paper for NCutLoss2D.
        adopted from https://github.com/fkodom/wnet-unsupervised-image-segmentation
    """

    def __init__(self, device : str, config : Dict[str,Any]) -> None:
        super().__init__(
            device=device, 
            config=config
        )
        #       radius: int = 2

    def forward(self, labels: torch.Tensor, **kwargs) -> torch.Tensor:
        """
            Computes the Opening loss -- i.e. the MSE due to performing a greyscale opening operation.

            :param labels: Predicted class probabilities
            :param args: Extra inputs, in case user also provides input/output image values.
            :return: Opening loss
        """

        smooth_labels = labels.clone().detach().cpu().detach().numpy()
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                smooth_labels[i, j] = grey_opening(
                    smooth_labels[i, j], self.radius)

        smooth_labels = torch.from_numpy(smooth_labels.astype(np.float32))
        if labels.device.type == 'cuda':
            smooth_labels = smooth_labels.cuda()

        return nn.MSELoss()(labels, smooth_labels.detach())
