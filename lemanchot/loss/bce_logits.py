
""" 
    @project LeManchot-Analysis : Core components
    @organization Laval University
    @lab MiViM Lab
    @supervisor Professor Xavier Maldague
    @industrial-partner TORNGATS
"""

from segmentation_models_pytorch.losses import SoftBCEWithLogitsLoss

from lemanchot.core import get_device
from lemanchot.loss.core import (
    BaseLoss, 
    classmap_2_multilayer, 
    loss_register
)

@loss_register('soft_bce')
class BCEWithLogitsLoss(BaseLoss):
    """
    Loss class for using SoftBCEWithLogitsLoss.
    Drop-in replacement for torch.nn.BCEWithLogitsLoss with few additions: ignore_index and label_smoothing
    """
    def __init__(self, name : str, config) -> None:
        """
        Args:
            name (str): The name of this dataset
            config (_type_): the configuration to initialize the loss class
        """
        super().__init__(name=name, config=config)
        config.pop('number_classes')
        self.criteria = SoftBCEWithLogitsLoss(**config).to(get_device())

    def prepare_loss(self, **kwargs):
        return

    def forward(self, output, target, **kwargs):
        trg = classmap_2_multilayer(target, self.number_classes).to(device=output.device)
        return self.criteria(output, trg)