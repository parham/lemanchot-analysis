

import logging
from dotmap import DotMap
from typing import Any, Dict, List, Union

from lemanchot.core import BaseCore, get_device

class BaseLoss(BaseCore):
    def __init__(self, name : str, config) -> None:
        super().__init__(
            name=name,
            config=config
        )
        
    def prepare_loss(self, **kwargs):
        return

    def forward(self, output, target, **kwargs):
        super().forward(output, target, **kwargs)

__loss_handler = {}

def loss_register(name : Union[str, List[str]]):
    def __embed_clss(clss):
        global __loss_handler
        hname = name if isinstance(name, list) else [name]
        if not issubclass(clss, BaseLoss):
            raise NotImplementedError('The specified loss handler is not correctly implemented!')
        for n in hname:
            __loss_handler[n] = clss

    return __embed_clss

def list_losses() -> List[str]:
    global __loss_handler
    return list(__loss_handler.keys())

def load_loss(loss_name : str, device : str, config : Dict[str,Any]):
    if not loss_name in list_losses():
        msg = f'{loss_name} model is not supported!'
        logging.error(msg)
        raise ValueError(msg)
    
    return __loss_handler[loss_name](loss_name, device, config)

def load_loss(experiment_config : DotMap):
    """Load an instance of a registered model based on the given name

    Args:
        experiment_config (DotMap): configuration

    Raises:
        ValueError: model is not supported

    Returns:
        BaseModule: the instance of the given model
    """

    # Get model name
    loss_name = experiment_config.name
    # Get the experiment configuration
    loss_config = experiment_config.config

    if not loss_name in list_losses():
        msg = f'{loss_name} model is not supported!'
        logging.error(msg)
        raise ValueError(msg)
    
    return __loss_handler[loss_name](
        name=loss_name,
        config=loss_config
    )

