
import torch
from torch import nn

from lemanchot.loss.core import BaseLoss, loss_register

@loss_register('cross_entropy')
class CrossEntropyLoss(BaseLoss):
    def __init__(self, name : str, config) -> None:
        super().__init__(name=name, config=config)
        self.criteria = nn.CrossEntropyLoss(reduction=self.reduction,ignore_index=self.ignore_index)

    def prepare_loss(self, **kwargs):
        return
    
    def forward(self, output, target, **kwargs):
        return self.criteria(output, target.squeeze(1))