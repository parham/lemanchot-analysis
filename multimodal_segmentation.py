
""" 
    @title A Deep Semi-Supervised Segmentation Approach for Thermographic Analysis of Industrial Components
    @organization Laval University
    @professor  Professor Xavier Maldague
    @author     Parham Nooralishahi
    @email      parham.nooralishahi@gmail.com
"""

import argparse
import os
import sys
import logging
from typing import List
import numpy as np

import cv2
from PIL import Image
from pathlib import Path
from tqdm import trange
from scipy.io import savemat, loadmat

import torch
import torchvision.transforms as transforms

from lemanchot.core import get_config, get_device
from lemanchot.loss.core import load_loss
from lemanchot.models.core import load_model
from lemanchot.pipeline.core import load_optimizer
from lemanchot.transform import (
    FilterOutAlphaChannel,
    ImageResize,
    ImageResizeByCoefficient,
    NumpyImageToTensor,
    ToFloatTensor
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

known_args = []
parser = argparse.ArgumentParser(description="Multi-Modal Analysis")
parser.add_argument('file', type=str, help="Multi-Modal file")
parser.add_argument('--iteration', type=int, default=60,
                    help="The maximum number of iteration.")
parser.add_argument('--nclass', type=int, default=10,
                    help="The minimum number of classes.")
# parser.add_argument('--output', type=str, default='.', help="The result folder.")
# parser.add_argument('--registeration', action='store_false', help="Enable the multi-modal registration feature.")

def region_segmentation():
    args = parser.parse_intermixed_args()
    parser.print_help()

    if not os.path.isfile(args.file):
        logging.error(f'{args.file} does not exist!')
        return

    device = get_device()

    data = loadmat(args.file)
    thermal = data['aligned_ir'] if 'aligned_ir' in data else None
    thermal_roi = data['ir_roi'] if 'ir_roi' in data else None
    visible = data['viz'] if 'viz' in data else None
    visible_roi = data['viz_roi'] if 'viz_roi' in data else None

    experiment_config = get_config('wonjik2020')
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
    transform = transforms.Compose([
        # ImageResize(70),
        ImageResizeByCoefficient(32),
        NumpyImageToTensor(),
        FilterOutAlphaChannel(),
        ToFloatTensor()
    ])
    # Apply the transformations to the given data
    input = transform(thermal_roi)
    input = input.to(dtype=torch.float32, device=device)

    criterion.prepare_loss(ref=input)

    result = None
    for iter in trange(args.iteration):
        model.train()
        optimizer.zero_grad()

        output = model(input.unsqueeze(0))
        output = output.squeeze(0)

        _, trg = torch.max(output, 0)
        loss = criterion(output, trg)
        trg = trg.unsqueeze(0).unsqueeze(0).to(dtype=torch.uint8)

        loss.backward()
        optimizer.step()

        num_classes = len(torch.unique(trg))
        if num_classes <= args.nclass:
            break

        result = trg

    logging.info('Saving the result ... ')
    thermal_seg = result.squeeze(0).squeeze(0)
    data['ir_seg'] = thermal_seg.cpu().detach().numpy()

    savemat(args.file, data, do_compression=True)


if __name__ == "__main__":
    region_segmentation()
