
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

@pipeline_register("simple_train")
def simple_train_step__(
    engine : Engine,
    batch,
    device,
    model : BaseModule,
    criterion,
    optimizer : optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    experiment : Experiment
) -> Dict:

    inputs, targets = batch

    inputs = inputs.to(dtype=torch.float32, device=device)
    targets = targets.to(dtype=torch.float32, device=device)

    criterion.prepare_loss(ref=batch[0])

    model.train()
    optimizer.zero_grad()

    outputs = model(inputs)

    loss = criterion(outputs, targets)
    # Threshold < thresh = value
    outputs = torch.threshold(outputs.sigmoid(), 0.5, 0)

    loss.backward()
    optimizer.step()
    scheduler.step(loss.item())

    return {
        'y' : batch[1],
        'y_pred' : outputs,
        'loss' : loss.item(),
        'lr' : optimizer.param_groups[0]['lr']
    }
