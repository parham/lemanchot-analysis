
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
from skimage import segmentation

from lemanchot.loss.core import BaseLoss, loss_register

@loss_register('unsupervised_superres_loss')
class UnsupervisedLoss_SuperResolusion(BaseLoss):
    """Loss function implemented based on the loss function defined in,
    @name           Unsupervised Image Segmentation by Backpropagation
    @journal        IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)
    @year           2018
    @repo           https://github.com/kanezaki/pytorch-unsupervised-segmentation
    @citation       Asako Kanezaki. Unsupervised Image Segmentation by Backpropagation. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2018.
    """

    def __init__(self, name : str, config) -> None:
        """
        Args:
            device (str): _description_
            config (Dict[str,Any]): _description_
                Defaults:
                    compactness: int = 100
                    superpixel_regions: int = 30
        """
        super().__init__(name=name, config=config)

        self.l_inds = None
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.to_pil = ToPILImage()

    def prepare_loss(self, **kwargs):
        """Set the reference image for SLIC algorithm duing initialization.

        Args:
            ref (Image): Reference image
        """
        ref = np.asarray(self.to_pil(kwargs['ref'].squeeze(0)))
        self._ref = ref
        img_h = ref.shape[0]
        img_w = ref.shape[1]
        # SLIC : segment the image using SLIC algorithm
        labels = segmentation.slic(ref,
            compactness=self.compactness,
            n_segments=self.superpixel_regions)
        # Flatten the resulted segmentation using SLIC
        labels = labels.reshape(img_w * img_h)
        # Extract the unique label
        u_labels = np.unique(labels)
        # Form the label indexes
        self.l_inds = []
        for i in range(len(u_labels)):
            self.l_inds.append(np.where(labels == u_labels[i])[0])

    def forward(self, output, target, **kwargs):
        # Superpixel Refinement
        im_target = target.cpu().detach().numpy()
        for i in range(len(self.l_inds)):
            labels_per_sp = im_target[self.l_inds[i]]
            u_labels_per_sp = np.unique(labels_per_sp)
            hist = np.zeros(len(u_labels_per_sp))
            for j in range(len(hist)):
                hist[j] = len(np.where(labels_per_sp == u_labels_per_sp[j])[0])
            im_target[self.l_inds[i]] = u_labels_per_sp[np.argmax(hist)]
        
        target_new = Variable(torch.from_numpy(im_target).to(self.device))

        return self.loss_fn(output, target_new)
