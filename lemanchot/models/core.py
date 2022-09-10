
""" 
    @project LeManchot-Analysis : Core components
    @organization Laval University
    @lab MiViM Lab
    @supervisor Professor Xavier Maldague
    @industrial-partner TORNGATS
"""

import logging

from dotmap import DotMap
from typing import List, Union

from lemanchot.core import BaseCore

class BaseModule(BaseCore):
    """ Base class for all modules """
    def __init__(self, name : str, config : DotMap) -> None:
        super().__init__(
            name=name,
            config=config
        )

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

def load_model(experiment_config : DotMap) -> BaseModule:
    """Load an instance of a registered model based on the given name

    Args:
        experiment_config (DotMap): configuration

    Raises:
        ValueError: model is not supported

    Returns:
        BaseModule: the instance of the given model
    """

    global __model_handler
    if not 'model' in experiment_config:
        return None

    # Get model name
    model_name = experiment_config.model.name
    # Get the model configuration
    model_config = experiment_config.model.config if 'config' in experiment_config.model else {}

    if not model_name in list_models():
        msg = f'{model_name} model is not supported!'
        logging.error(msg)
        raise ValueError(msg)
    
    return __model_handler[model_name](
        name=model_name,
        config=model_config
    )

