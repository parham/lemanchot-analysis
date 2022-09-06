
""" 
    @project LeManchot : Multi-Modal Data Acquisition and Processing of Drone-based Inspection
    @organization Laval University
    @lab MiViM Lab
    @supervisor Professor Xavier Maldague
    @industrial-partner TORNGATS
"""

from dotmap import DotMap

import segmentation_models_pytorch as smp

from lemanchot.models.core import BaseModule, model_register

@model_register('unet_resnet18')
class Unet_Resnet18(BaseModule):
    def __init__(self, name : str, config : DotMap) -> None:
        super().__init__(
            name='unet_resnet18',
            config=config
        )
        self.clss = smp.Unet(
            encoder_name='resnet18',
            encoder_weights="imagenet",
            in_channels=self.channels,
            classes=self.num_classes 
        )
    
    def forward(self, x):
        return self.clss(x)

@model_register('unetplusplus_resnet18')
class UnetPlusPlus_Resnet18(BaseModule):
    def __init__(self, name : str, config : DotMap) -> None:
        super().__init__(
            name='unetplusplus_resnet18',
            config=config
        )
        self.clss = smp.UnetPlusPlus(
            encoder_name='resnet18',
            encoder_weights="imagenet",
            in_channels=self.channels,
            classes=self.num_classes 
        )

    def forward(self, x):
        return self.clss(x)