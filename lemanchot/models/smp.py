"""
    @project LeManchot-Analysis : Core components
    @organization Laval University
    @lab MiViM Lab
    @supervisor Professor Xavier Maldague
    @industrial-partner TORNGATS
"""

from dotmap import DotMap

import segmentation_models_pytorch as smp

from lemanchot.core import get_or_default
from lemanchot.models.core import BaseModule, model_register


@model_register("unet_densenet121")
class Unet_Resnet18(BaseModule):
    """Implementation of SMP UNET DENSENET-121"""

    def __init__(self, name: str, config: DotMap) -> None:
        super().__init__(name="unet_densenet121", config=config)
        self.clss = smp.Unet(
            encoder_name="densenet121",
            encoder_weights=get_or_default(config, "weights", None),
            in_channels=get_or_default(config, "channels", 3),
            classes=get_or_default(config, "num_classes", 2),
            activation=get_or_default(config, "activation", None),
        )

    def forward(self, x):
        return self.clss(x)


@model_register("unet_resnet18")
class Unet_Resnet18(BaseModule):
    """Implementation of SMP UNET RESNET-18"""

    def __init__(self, name: str, config: DotMap) -> None:
        super().__init__(name="unet_resnet18", config=config)
        self.clss = smp.Unet(
            encoder_name="resnet18",
            encoder_weights=get_or_default(config, "weights", None),
            in_channels=get_or_default(config, "channels", 3),
            classes=get_or_default(config, "num_classes", 2),
            activation=get_or_default(config, "activation", None),
        )

    def forward(self, x):
        return self.clss(x)


@model_register("unet_resnet50")
class Unet_Resnet50(BaseModule):
    """Implementation of SMP UNET RESNET-50"""

    def __init__(self, name: str, config: DotMap) -> None:
        super().__init__(name="unet_resnet50", config=config)
        self.clss = smp.Unet(
            encoder_name="resnet50",
            encoder_weights=get_or_default(config, "weights", None),
            in_channels=get_or_default(config, "channels", 3),
            classes=get_or_default(config, "num_classes", 2),
            activation=get_or_default(config, "activation", None),
        )

    def forward(self, x):
        return self.clss(x)


@model_register("unetplusplus_resnet18")
class UnetPlusPlus_Resnet18(BaseModule):
    """Implementation of SMP UNET++ RESNET-18"""

    def __init__(self, name: str, config: DotMap) -> None:
        super().__init__(name="unetplusplus_resnet18", config=config)
        self.clss = smp.UnetPlusPlus(
            encoder_name="resnet18",
            encoder_weights=get_or_default(config, "weights", None),
            in_channels=get_or_default(config, "channels", 3),
            classes=get_or_default(config, "num_classes", 2),
            activation=get_or_default(config, "activation", None),
        )

    def forward(self, x):
        return self.clss(x)


@model_register("unetplusplus_resnet50")
class UnetPlusPlus_Resnet50(BaseModule):
    """Implementation of SMP UNET++ RESNET-50"""

    def __init__(self, name: str, config: DotMap) -> None:
        super().__init__(name="unetplusplus_resnet50", config=config)
        self.clss = smp.UnetPlusPlus(
            encoder_name="resnet50",
            encoder_weights=get_or_default(config, "weights", None),
            in_channels=get_or_default(config, "channels", 3),
            classes=get_or_default(config, "num_classes", 2),
            activation=get_or_default(config, "activation", None),
        )

    def forward(self, x):
        return self.clss(x)


@model_register("fpn_resnet18")
class FPN_Resnet18(BaseModule):
    """Implementation of SMP FPN RESNET-18"""

    def __init__(self, name: str, config: DotMap) -> None:
        super().__init__(name="fpn_resnet18", config=config)
        self.clss = smp.FPN(
            encoder_name="resnet18",
            encoder_weights=get_or_default(config, "weights", None),
            in_channels=get_or_default(config, "channels", 3),
            classes=get_or_default(config, "num_classes", 2),
            activation=get_or_default(config, "activation", None),
        )

    def forward(self, x):
        return self.clss(x)


@model_register("fpn_resnet50")
class FPN_Resnet18(BaseModule):
    """Implementation of SMP FPN RESNET-50"""

    def __init__(self, name: str, config: DotMap) -> None:
        super().__init__(name="fpn_resnet50", config=config)
        self.clss = smp.FPN(
            encoder_name="resnet50",
            encoder_weights=get_or_default(config, "weights", None),
            in_channels=get_or_default(config, "channels", 3),
            classes=get_or_default(config, "num_classes", 2),
            activation=get_or_default(config, "activation", None),
        )

    def forward(self, x):
        return self.clss(x)


@model_register("deeplabv3plus_resnet50")
class DeeplabV3Plus_Resnet50(BaseModule):
    """Implementation of SMP DeepLabV3+ RESNET-50"""

    def __init__(self, name: str, config: DotMap) -> None:
        super().__init__(name="deeplabv3plus_resnet50", config=config)
        self.clss = smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights=get_or_default(config, "weights", None),
            in_channels=get_or_default(config, "channels", 3),
            classes=get_or_default(config, "num_classes", 2),
            activation=get_or_default(config, "activation", None),
        )

    def forward(self, x):
        return self.clss(x)
