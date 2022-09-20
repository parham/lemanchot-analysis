
"""
    @project LeManchot-Analysis : Core components
    @organization Laval University
    @lab MiViM Lab
    @supervisor Professor Xavier Maldague
    @industrial-partner TORNGATS
"""

from comet_ml import Experiment
from typing import Dict

import torch.optim as optim
from torch import no_grad, threshold
from ignite.engine import Engine

from lemanchot.pipeline.core import pipeline_register
from lemanchot.models import BaseModule

@pipeline_register("simple_train")
def simple_train_step__(
    engine : Engine,
    batch,
    device,
    model : BaseModule,
    criterion,
    optimizer : optim.Optimizer,
    experiment : Experiment
) -> Dict:

    inputs, targets = batch

    criterion.prepare_loss(ref=batch[0])

    model.train()
    optimizer.zero_grad()

    outputs = model(inputs)

    loss = criterion(outputs, targets)
    outputs = outputs.argmax(dim=1, keepdims=True)

    loss.backward()
    optimizer.step()

    return {
        'y_true' : targets,
        'y_pred' : outputs,
        'loss' : loss.item()
    }

@pipeline_register("simple_multilabel")
def simple_multilabel_step__(
    engine : Engine,
    batch,
    device,
    model : BaseModule,
    criterion,
    optimizer : optim.Optimizer,
    experiment : Experiment
) -> Dict:

    inputs, targets = batch

    criterion.prepare_loss(ref=batch[0])

    model.train()
    optimizer.zero_grad()

    outputs = model(inputs)

    loss = criterion(outputs, targets)

    loss.backward()
    optimizer.step()
    
    outputs = threshold(outputs.sigmoid(), 0.5, 0)

    return {
        'y_true' : targets,
        'y_pred' : outputs,
        'loss' : loss.item()
    }