
""" 
    @organization Laval University
    @professor  Professor Xavier Maldague
    @author     Parham Nooralishahi
    @email      parham.nooralishahi@gmail.com
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
    experiment : Experiment
) -> Dict:

    inputs, targets = batch
    
    inputs = inputs.to(dtype=torch.float32, device=device)
    targets = targets.to(dtype=torch.float32, device=device)

    criterion.prepare_loss(ref=batch[0])

    model.train()
    optimizer.zero_grad()

    outputs = model(inputs)

    outputs = torch.tensor(torch.argmax(outputs, dim=1), dtype=targets.dtype, requires_grad=True)
    targets = targets.squeeze(1)
    loss = criterion(outputs, targets)

    loss.backward()
    optimizer.step()

    return {
        'y' : targets,
        'y_pred' : outputs,
        'loss' : loss.item()
    }