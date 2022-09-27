"""
    @project LeManchot-Analysis : Core components
    @organization Laval University
    @lab MiViM Lab
    @supervisor Professor Xavier Maldague
    @industrial-partner TORNGATS
"""

from typing import Dict

import torch
from ignite.engine import Engine

from lemanchot.pipeline.core import pipeline_register
from lemanchot.models import BaseModule


@pipeline_register("simple_predict")
@torch.no_grad()
def simple_predict_step__(engine: Engine, batch, model: BaseModule, **kwargs) -> Dict:

    inputs = batch

    model.eval()
    outputs = model(inputs)
    outputs = outputs.argmax(dim=1, keepdims=True)

    return {
        'y_pred' : outputs
    }


@pipeline_register("simple_pavement_predict")
@torch.no_grad()
def simple_multilabel_step__(
    engine: Engine, batch, model: BaseModule, **kwargs
) -> Dict:

    inputs = batch

    model.eval()
    outputs = model(inputs)
    outputs = torch.threshold(outputs.sigmoid(), 0.5, 0)

    return {
        'y_pred': outputs
    }
