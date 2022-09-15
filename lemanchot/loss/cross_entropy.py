
""" 
    @project LeManchot-Analysis : Core components
    @organization Laval University
    @lab MiViM Lab
    @supervisor Professor Xavier Maldague
    @industrial-partner TORNGATS
"""

from torch import nn

from lemanchot.loss.core import BaseLoss, loss_register

@loss_register('cross_entropy')
class CrossEntropyLoss(BaseLoss):
    def __init__(self, name : str, config) -> None:
        super().__init__(name=name, config=config)
        self.criteria = nn.BCEWithLogitsLoss(**config).to(self.device)

    def prepare_loss(self, **kwargs):
        return
    
    def forward(self, output, target, **kwargs):
        return self.criteria(output, target.float())