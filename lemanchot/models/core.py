


""" 
    @title        Multi-modal texture analysis to enhance drone-based thermographic inspection of structures 
    @organization Laval University
    @partner      TORNGATS
"""
import logging

from dotmap import DotMap
from typing import List, Union

from lemanchot.core import BaseCore, get_device

__model_handler = {}

def model_register(name : Union[str, List[str]]):
    """Register a model into the model repository

    Args:
        name (Union[str, List[str]]): the model's or list of model's name(s)
    """
    def __embed_func(clss):
        global __model_handler
        hname = name if isinstance(name, list) else [name]
        if not issubclass(clss, BaseModule):
            raise NotImplementedError('The specified model is not correctly implemented!')

        for n in hname:
            __model_handler[n] = clss

    return __embed_func

def list_models() -> List[str]:
    """List of registered models

    Returns:
        List[str]: list of registered models
    """
    global __model_handler
    return list(__model_handler.keys())

def load_model(experiment_config : DotMap):
    """Load an instance of a registered model based on the given name

    Args:
        model_name (str): model's name
        device (str): device's name
        config (_type_): configuration

    Raises:
        ValueError: model is not supported

    Returns:
        BaseModule: the instance of the given model
    """

    # Get device name
    device = get_device()
    # Get model name
    model_name = experiment_config.name
    # Get the experiment configuration
    model_config = experiment_config.config

    if not model_name in list_models():
        msg = f'{model_name} model is not supported!'
        logging.error(msg)
        raise ValueError(msg)
    
    return __model_handler[model_name](
        name=model_name, 
        device=device, 
        config=model_config
    )

class BaseModule(BaseCore):
    """ Base class for all modules """
    def __init__(self, name : str, device : str, config : DotMap) -> None:
        super().__init__(
            name=name,
            device=device,
            config=config
        )