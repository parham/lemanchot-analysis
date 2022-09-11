
""" 
    @title A Deep Semi-supervised Segmentation Approach for Thermographic Analysis of Industrial Components
    @organization Laval University
    @professor  Professor Xavier Maldague
    @author     Parham Nooralishahi
    @email      parham.nooralishahi@gmail.com
"""

import torch
from torch.autograd import Variable
from torchvision.transforms import ToPILImage

import numpy as np

from lemanchot.loss.core import BaseLoss, loss_register

@loss_register('unsupervised_twofactor_loss')
class UnsupervisedLoss_TwoFactors(BaseLoss):
    """ Loss function implemented based on the loss function defined in,
    @name           Unsupervised Learning of Image Segmentation Based on Differentiable Feature Clustering   
    @journal        IEEE Transactions on Image Processing
    @year           2020
    @repo           https://github.com/kanezaki/pytorch-unsupervised-segmentation-tip
    @citation       Wonjik Kim*, Asako Kanezaki*, and Masayuki Tanaka. Unsupervised Learning of Image Segmentation Based on Differentiable Feature Clustering. IEEE Transactions on Image Processing, accepted, 2020.
    """

    def __init__(self, name : str, config) -> None:
        """
        Args:
            name (str): _description_
            config (_type_): _description_
                Defaults:
                    num_channel: int = 100,
                    similarity_loss: float = 1.0,
                    continuity_loss: float = 0.5
        """

        super().__init__(name=name, config=config)

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.loss_hpy = torch.nn.L1Loss(reduction='mean')
        self.loss_hpz = torch.nn.L1Loss(reduction='mean')
        self.HPy_target = None
        self.HPz_target = None
        self.to_pil = ToPILImage()

    def prepare_loss(self, **kwargs):
        ref = np.asarray(self.to_pil(kwargs['ref'].squeeze(0)))
        self._ref = ref
        img_h = ref.shape[0]
        img_w = ref.shape[1]
        self.HPy_target = torch.zeros(
            self.num_channels, img_h - 1, img_w).to(self.device)
        self.HPz_target = torch.zeros(
            self.num_channels, img_h, img_w - 1).to(self.device)

    def forward(self, output, target, **kwargs):
        HPy = output[:, 1:, :] - output[:, 0:-1, :]
        HPz = output[:, :, 1:] - output[:, :, 0:-1]
        lhpy = self.loss_hpy(HPy, self.HPy_target)
        lhpz = self.loss_hpz(HPz, self.HPz_target)
        # loss calculation
        return self.similarity_loss * \
            self.loss_fn(output.unsqueeze(dim=0), target.unsqueeze(dim=0)) + \
            self.continuity_loss * (lhpy + lhpz)