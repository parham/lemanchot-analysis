
""" 
    @title A Deep Semi-supervised Segmentation Approach for Thermographic Analysis of Industrial Components
    @organization Laval University
    @professor  Professor Xavier Maldague
    @author     Parham Nooralishahi
    @email      parham.nooralishahi@gmail.com
"""

import time
from typing import Dict
from ignite.engine import Engine

import torch
import torch.optim as optim
from torchvision.transforms import ToPILImage, Grayscale, ToTensor

import numpy as np
from sklearn.cluster import DBSCAN, KMeans, MeanShift, estimate_bandwidth

from comet_ml import Experiment

from lemanchot.pipeline.core import pipeline_register
from lemanchot.models import BaseModule
from lemanchot.processing import adapt_output

@pipeline_register("classics")
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

    to_pil = ToPILImage()
    to_tensor = ToTensor()

    inputs = np.asarray((to_pil(inputs.squeeze(0))))
    targets = np.asarray((to_pil(targets.squeeze(0))))
    # Pass data to the model
    outputs = model(inputs)
    postprocessed = adapt_output(outputs, targets, iou_thresh=engine.state.iou_thresh)

    outputs = to_tensor(outputs).unsqueeze(0)
    postprocessed = to_tensor(postprocessed[0]).unsqueeze(0)

    return {
        'y' : batch[1],
        'y_pred' : outputs,
        'y_processed' : postprocessed,
        'loss' : 0.001
    }

