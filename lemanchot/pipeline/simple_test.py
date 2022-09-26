"""
    @project LeManchot-Analysis : Core components
    @organization Laval University
    @lab MiViM Lab
    @supervisor Professor Xavier Maldague
    @industrial-partner TORNGATS
"""

from comet_ml import Experiment
from typing import Dict

import torch
import torch.optim as optim
from ignite.engine import Engine

from lemanchot.pipeline.core import pipeline_register
from lemanchot.models import BaseModule


@pipeline_register("simple_test")
@torch.no_grad()
def simple_train_step__(
    engine: Engine,
    batch,
    model: BaseModule,
    criterion,
    **kwargs
) -> Dict:

    inputs, targets = batch

    criterion.prepare_loss(ref=batch[0])

    model.eval()
    outputs = model(inputs)

    loss = criterion(outputs, targets)
    outputs = outputs.argmax(dim=1, keepdims=True)

    if len(targets.shape) < 4:
        targets = targets.unsqueeze(1)

    return {"y_true": targets, "y_pred": outputs, "loss": loss.item()}


@torch.no_grad()
@pipeline_register("simple_multilabel_test")
def simple_multilabel_step__(
    engine: Engine,
    batch,
    model: BaseModule,
    criterion,
    **kwargs
) -> Dict:

    inputs, targets = batch

    criterion.prepare_loss(ref=batch[0])

    model.eval()

    outputs = model(inputs)
    loss = criterion(outputs, targets)

    outputs = torch.threshold(outputs.sigmoid(), 0.5, 0)

    return {"y_true": targets, "y_pred": outputs, "loss": loss.item()}
