

import logging
import time
from typing import Any, Callable, Dict, List, Union
from uuid import uuid4

import torch.optim as optim
from comet_ml import Experiment

from ignite.engine import Engine
from ignite.handlers import global_step_from_engine

from lemanchot.core import (
    exception_logger,
    get_or_default,
    get_device, 
    get_profile, 
    make_tensor_for_comet
)
from lemanchot.loss.core import BaseLoss
from lemanchot.models.core import BaseModule
from lemanchot.metrics.core import BaseMetric

__wrapper_handler = {}

class BaseWrapper:
    def __init__(self,
        step_func : Callable,
        device : str,
        **kwargs
    ) -> None:
        super().__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.step_func = step_func
        self.device = device
    
    def __call__(self, engine : Engine, batch) -> Dict:
        pass

def wrapper_register(name : Union[str, List[str]]):
    """Register a wrapper into the wrapper repository
    """
    def __embed_func(clss):
        global __wrapper_handler
        hname = name if isinstance(name, list) else [name]
        if not issubclass(clss, BaseWrapper):
            raise NotImplementedError('The specified wrapper is not correctly implemented!')

        for n in hname:
            __wrapper_handler[n] = clss

    return __embed_func

def list_wrappers() -> List[str]:
    """List of registered wrappers

    Returns:
        List[str]: list of registered wrappers
    """
    global __wrapper_handler
    return list(__wrapper_handler.keys())

@exception_logger
def load_wrapper(
    wrapper_name : str,
    step_func : Callable,
    device : str,
    **kwargs
) -> BaseWrapper:
    """Load an instance of a registered wrapper based on the given name

    Args:
        experiment_config (DotMap): configuration

    Raises:
        ValueError: wrapper is not supported

    Returns:
        BaseModule: the instance of the given wrapper
    """

    global __wrapper_handler

    if not wrapper_name in list_wrappers():
        msg = f'{wrapper_name} model is not supported!'
        logging.error(msg)
        raise ValueError(msg)
    
    return __wrapper_handler[wrapper_name](
        step_func,
        device,
        **kwargs
    )

@wrapper_register('default_pipeline')
class DefaultWrapper(BaseWrapper):
    def __init__(self, 
        step_func: Callable, 
        device: str, 
        **kwargs
    ) -> None:
        super().__init__(step_func, device, **kwargs)
    
    def __call__(self,
        engine: Engine,
        batch
    ) -> Dict:

        # Local Variable Initialization
        step_func = self.step_func
        device = self.device
        model = self.model
        loss = self.loss
        optimizer = self.optimizer
        metrics = self.metrics
        experiment = self.experiment
        img_saver = self.img_saver if hasattr(self, 'img_saver') else None

        profile = get_profile(engine.state.profile_name)

        data = list(map(lambda x: x.to(device=get_device()), batch[0:2]))
        # Logging computation time
        t = time.time()
        # Apply the model to data
        res = step_func(
            engine=engine,
            batch=data,
            device=device,
            model=model,
            criterion=loss,
            optimizer=optimizer,
            experiment=experiment,
        )
        step_time = time.time() - t

        # Logging loss & step time
        if "loss" in res:
            engine.state.metrics["loss"] = res["loss"]
        engine.state.metrics["step_time"] = step_time

        targets = res["y_true"]
        outputs = res["y_pred"] if not "y_processed" in res else res["y_processed"]

        # Calculate metrics
        for m in metrics:
            m.update((outputs, targets))
            m.compute(engine, experiment)

        if profile.enable_logging:
            # Calculate metrics
            if "metrics" in res:
                engine.state.metrics.update(res["metrics"])

            # Assume Tensor B x C x W x H
            # Logging imagery results
            for key, img in res.items():
                if not "y_" in key:
                    continue
                # Control number of logged images with enable_image_logging setting.
                for i in range(min(profile.enable_image_logging, img.shape[0])):
                    sample = make_tensor_for_comet(img[i, :, :, :])
                    label = f"{key}-{engine.state.epoch}-{i}"
                    experiment.log_image(sample, label, step=engine.state.iteration)
        
                    if img_saver is not None and key == "y_pred":
                        img_saver(label, sample)
        
        return res
