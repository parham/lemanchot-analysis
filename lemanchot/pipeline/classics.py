
""" 
    @title A Deep Semi-supervised Segmentation Approach for Thermographic Analysis of Industrial Components
    @organization Laval University
    @professor  Professor Xavier Maldague
    @author     Parham Nooralishahi
    @email      parham.nooralishahi@gmail.com
"""

from time import time
from typing import Dict
from ignite.engine import Engine

import torch
import torch.optim as optim

import numpy as np
from sklearn.cluster import DBSCAN, KMeans, MeanShift, estimate_bandwidth

from comet_ml import Experiment

from lemanchot.pipeline.core import pipeline_register
from lemanchot.models import BaseModule

@pipeline_register("dbscan")
def dbscan_train_step__(
    engine : Engine, 
    batch,
    device,
    model : BaseModule,
    criterion,
    optimizer : optim.Optimizer,
    experiment : Experiment
) -> Dict:
    inputs, targets = batch

    t = time.time()

    inputs = inputs.cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()
    batch_size = inputs.shape[0]

    criterion.prepare_loss(ref=batch[0])

    outputs = model(inputs)

    outputs = torch.tensor(torch.argmax(outputs, dim=1), dtype=targets.dtype, requires_grad=True)
    targets = targets.squeeze(1)
    loss = criterion(outputs, targets)

    return {
        'y' : targets,
        'y_pred' : outputs,
        'loss' : loss.item()
    }

