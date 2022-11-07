"""
    @project LeManchot-Analysis : Core components
    @organization Laval University
    @lab MiViM Lab
    @supervisor Professor Xavier Maldague
    @industrial-partner TORNGATS
"""
import json
import os
import time
from itertools import product
from typing import Callable, Dict, List, NoReturn, Optional
from typing import T as Array
from typing import Tuple

import numpy as np
import torch
from comet_ml import Experiment
from ignite.engine import Engine
from torchvision.transforms.functional import resize

from lemanchot.core import make_tensor_for_comet
from lemanchot.models import BaseModule
from lemanchot.pipeline.core import get_device, get_profile, pipeline_register
from lemanchot.pipeline.wrapper import BaseWrapper, wrapper_register


@wrapper_register("default_validator")
class ValidatorWrapper(BaseWrapper):
    def __init__(self, step_func: Callable, device: str, **kwargs) -> None:
        super().__init__(step_func, device, **kwargs)

    def __call__(self, engine: Engine, batch) -> Dict:

        # Local Variable Initialization
        step_func = self.step_func
        device = self.device
        model = self.model
        loss = self.loss
        metrics = self.metrics
        experiment = self.experiment
        img_saver = self.img_saver if hasattr(self, "img_saver") else None

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


@pipeline_register("simple_multilabel_predict")
@torch.no_grad()
def simple_multilabel_predict__(
    engine: Engine, batch, device, model: BaseModule, criterion, experiment: Experiment
) -> Dict:
    inputs, targets = batch

    criterion.prepare_loss(ref=batch[0])

    model.eval()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    outputs = torch.where(outputs.sigmoid() > 0.5, 1, 0)

    return {"y_true": targets, "y_pred": outputs, "loss": loss.item()}
