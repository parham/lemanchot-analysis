
""" 
    @project LeManchot-Analysis : Core components
    @organization Laval University
    @lab MiViM Lab
    @supervisor Professor Xavier Maldague
    @industrial-partner TORNGATS
"""

import torch

import logging
from dotmap import DotMap
from typing import List, Union

from lemanchot.core import BaseCore

def classmap_2_multilayer(data, number_classes):
    """This method assumes that data is NxHxW

    Args:
        data (_type_): _description_
        number_classes (_type_): _description_
    """
    sz = data.shape
    ml_data = torch.zeros((sz[0], number_classes, sz[1], sz[2]))
    for i in range(sz[0]):
        for c in range(number_classes):
            tmp = torch.zeros(sz[1], sz[2])
            ctmp = data[i,:,:]
            tmp[ctmp == c] = 1.
            ml_data[i,c,:,:] = tmp
    return ml_data

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
    """Register the loss (decorator)

    Args:
        name (Union[str, List[str]]): the name of given loss
    """
    def __embed_clss(clss):
        global __loss_handler
        hname = name if isinstance(name, list) else [name]
        if not issubclass(clss, BaseLoss):
            raise NotImplementedError('The specified loss handler is not correctly implemented!')
        for n in hname:
            __loss_handler[n] = clss

    return __embed_clss

def list_losses() -> List[str]:
    """List of losses

    Returns:
        List[str]: list containing the names of registered losses
    """
    global __loss_handler
    return list(__loss_handler.keys())

def load_loss_inline__(name : str, config):
    global __loss_handler
    if not name in list_losses():
        msg = f'{name} model is not supported!'
        logging.error(msg)
        raise ValueError(msg)
    
    return __loss_handler[name](
        name=name,
        config=config
    )

def load_loss(experiment_config : DotMap):
    """Load an instance of a registered loss based on the given name

    Args:
        experiment_config (DotMap): configuration

    Raises:
        ValueError: loss is not supported

    Returns:
        BaseModule: the instance of the given loss
    """

    global __loss_handler
    if not 'loss' in experiment_config:
        return None

    # Get loss name
    loss_name = experiment_config.loss.name
    # Get the loss configuration
    loss_config = experiment_config.loss.config if 'config' in experiment_config.loss else {}

    if not loss_name in list_losses():
        msg = f'{loss_name} loss is not supported!'
        logging.error(msg)
        raise ValueError(msg)
    
    return __loss_handler[loss_name](
        name=loss_name,
        config=loss_config
    )
