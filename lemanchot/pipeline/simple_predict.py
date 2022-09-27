"""
    @project LeManchot-Analysis : Core components
    @organization Laval University
    @lab MiViM Lab
    @supervisor Professor Xavier Maldague
    @industrial-partner TORNGATS
"""

from typing import Dict, Callable, NoReturn
import time
import torch
from ignite.engine import Engine

from lemanchot.pipeline.core import (
    pipeline_register,
    get_device,
    get_profile,
    make_tensor_for_comet,
)
from lemanchot.models import BaseModule
from .wrapper import BaseWrapper, wrapper_register


@wrapper_register("default_predict")
class PredictWrapper(BaseWrapper):
    def __init__(self, step_func: Callable, device: str, **kwargs) -> None:
        super().__init__(step_func, device, **kwargs)

    def __call__(self, engine: Engine, batch) -> NoReturn:

        # Local Variable Initialization
        step_func = self.step_func
        device = self.device
        model = self.model
        img_saver = self.img_saver if hasattr(self, "img_saver") else None

        profile = get_profile(engine.state.profile_name)
        paths, data = batch
        # Logging computation time
        t = time.time()
        # Apply the model to data
        res = step_func(
            engine=engine,
            batch=data.to(device=get_device()),
            device=device,
            model=model,
        )
        step_time = time.time() - t

        # Logging loss & step time
        if "loss" in res:
            engine.state.metrics["loss"] = res["loss"]
        engine.state.metrics["step_time"] = step_time

        if profile.enable_logging and img_saver is not None:
            # Assume Tensor B x C x W x H
            # Logging imagery results
            for name, img in zip(paths, res["y_pred"]):
                sample = make_tensor_for_comet(img)
                img_saver(name, sample)


@pipeline_register("simple_predict")
@torch.no_grad()
def simple_predict_step__(engine: Engine, batch, model: BaseModule, **kwargs) -> Dict:

    inputs = batch

    model.eval()
    outputs = model(inputs)
    outputs = outputs.argmax(dim=1, keepdims=True)

    return {"y_pred": outputs}


@pipeline_register("simple_multilabel_predict")
@torch.no_grad()
def simple_multilabel_step__(
    engine: Engine, batch, model: BaseModule, **kwargs
) -> Dict:

    inputs = batch

    model.eval()
    outputs = model(inputs)
    outputs = torch.threshold(outputs.sigmoid(), 0.5, 0)

    return {"y_pred": outputs}
