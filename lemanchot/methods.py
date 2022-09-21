
""" 
    @title A Deep Semi-supervised Segmentation Approach for Thermographic Analysis of Industrial Components
    @organization Laval University
    @professor  Professor Xavier Maldague
    @author     Parham Nooralishahi
    @email      parham.nooralishahi@gmail.com
"""

import logging
import torch

from lemanchot.core import get_device
from lemanchot.loss.core import load_loss
from lemanchot.models.core import load_model
from lemanchot.pipeline.core import load_optimizer


def iterative_region_segmentation(
    batch,
    experiment_config,
    device,
    num_iteration : int,
    class_count_limit : int
):
    input = batch[0]
    input = input.to(dtype=torch.float32, device=device)
    # Create model instance
    logging.info('Loading model ...')
    model = load_model(experiment_config)
    model.to(device)
    # Create loss instance
    logging.info('Loading loss ...')
    criterion = load_loss(experiment_config)
    criterion.to(device)
    # Create optimizer instance
    logging.info('Loading optimizer ...')
    optimizer = load_optimizer(model, experiment_config)
    # Create transformations
    logging.info('Creating and Applying transformations ...')
    # Apply the transformations to the given data
    input = input.to(dtype=torch.float32, device=device)
    # Prepare the loss function for the pipeline
    criterion.prepare_loss(ref=input)

    result = None
    for iter in range(num_iteration):
        # Set the model into the training mode
        model.train()
        optimizer.zero_grad()
        # Pass the data to the model
        output = model(input)
        output = output.squeeze(0)
        # Calculate the output as the new target
        _, trg = torch.max(output, 0)
        loss = criterion(output, trg)
        trg = trg.unsqueeze(0).unsqueeze(0).to(dtype=torch.uint8)

        loss.backward()
        optimizer.step()
        # Calculate the number of calculated classes
        num_classes = len(torch.unique(trg))
        # Check that the number of classes is not less than the defined limit
        if num_classes <= class_count_limit:
            break

        yield {
            'y' : batch[1],
            'y_pred' : trg,
            'loss' : loss.item(),
            'class_count' : num_classes
        }