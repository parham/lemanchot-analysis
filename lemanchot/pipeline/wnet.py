
""" 
    @title A Deep Semi-supervised Segmentation Approach for Thermographic Analysis of Industrial Components
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

@pipeline_register("wnet")
def wnet_train_step__(
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

    masks, outputs = model(inputs)
    inputs, targets, outputs = inputs.contiguous(), targets.contiguous(), outputs.contiguous()

    loss = criterion(outputs, targets, inputs, masks)
    outputs = torch.tensor(torch.argmax(outputs, dim=1), dtype=targets.dtype).unsqueeze(1)

    loss.backward()
    optimizer.step()

    return {
        'y' : batch[1],
        'y_pred' : outputs,
        'loss' : loss.item()
    }