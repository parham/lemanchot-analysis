
""" 
    @project LeManchot-Analysis : Core components
    @organization Laval University
    @lab MiViM Lab
    @supervisor Professor Xavier Maldague
    @industrial-partner TORNGATS
"""

import torch
from torch import nn

from segmentation_models_pytorch.losses import SoftBCEWithLogitsLoss, DiceLoss
from lemanchot.core import get_device

from lemanchot.loss.core import BaseLoss, classmap_2_multilayer, loss_register

@loss_register('soft_bce')
class SMP_BCEWithLogitsLoss(BaseLoss):
    """ The smp implementation of BCEWithLogitsLoss """
    def __init__(self, name : str, config) -> None:
        super().__init__(name=name, config=config)
        config.pop('number_classes')
        self.criteria = SoftBCEWithLogitsLoss(**config).to(get_device())

    def prepare_loss(self, **kwargs):
        return
    
    def forward(self, output, target, **kwargs):
        trg = classmap_2_multilayer(target, self.number_classes).to(device=output.device)
        return self.criteria(output, trg)

@loss_register('dice')
class SMP_DiceLoss(BaseLoss):
    """ The smp implementation of DiceLoss

        Example of configuration section
        {
            "mode" : "multiclass",
            "number_classes" : 7
        }
    """
    
    def __init__(self, name : str, config) -> None:
        super().__init__(name=name, config=config)
        config.pop('number_classes')
        self.criteria = DiceLoss(**config).to(get_device())

    def prepare_loss(self, **kwargs):
        return
    
    def forward(self, output, target, **kwargs):
        return self.criteria(output, target)