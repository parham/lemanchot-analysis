
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

from lemanchot.core import get_config
from lemanchot.loss.core import load_loss
from lemanchot.models.core import load_model
from lemanchot.pipeline.core import load_optimizer
from lemanchot.tools.control_point import cpselect
from lemanchot.transform import FilterOutAlphaChannel, ImageResize, ImageResizeByCoefficient, NumpyImageToTensor, ToFloatTensor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

known_args = []
parser = argparse.ArgumentParser(description="Multi-Modal Analysis")
parser.add_argument('file', type=str, help="Multi-Modal file")
parser.add_argument('--device', type=str, default='cuda', help="The selected device.")
parser.add_argument('--iteration', type=int, default=60, help="The maximum number of iteration.")
parser.add_argument('--nclass', type=int, default=10, help="The minimum number of classes.")
# parser.add_argument('--output', type=str, default='.', help="The result folder.")
# parser.add_argument('--registeration', action='store_false', help="Enable the multi-modal registration feature.")

def cp_to_opencv(cps : List):
    source = np.zeros((len(cps), 2))
    dest = np.zeros((len(cps), 2))
    for index in range(len(cps)):
        p = cps[index]
        source[index, 0] = p['img1_x']
        source[index, 1] = p['img1_y']
        dest[index, 0] = p['img2_x']
        dest[index, 1] = p['img2_y']
    return source, dest

def save_homography(file : str, homography : np.ndarray):
    mat = {
        'homography' : homography
    }
    savemat(file, mat, do_compression=True)

def region_segmentation():
    args = parser.parse_intermixed_args()
    parser.print_help()

    if not os.path.isfile(args.tfile):
        logging.error(f'{args.tfile} does not exist!')
        return
    
    data = loadmat(args.file)
    thermal = data['aligned_ir'] if 'aligned_ir' in data else None
    thermal_roi = data['ir_roi'] if 'ir_roi' in data else None
    visible = data['viz'] if 'viz' in data else None
    visible_roi = data['viz_roi'] if 'viz_roi' in data else None

    experiment_config = get_config('wonjik2020')
    # Create model instance
    logging.info('Loading model ...')
    model = load_model(experiment_config)
    model.to(args.device)
    # Create loss instance
    logging.info('Loading loss ...')
    criterion = load_loss(experiment_config)
    criterion.to(args.device)
    # Create optimizer instance
    logging.info('Loading optimizer ...')
    optimizer = load_optimizer(model, experiment_config)

    logging.info('Creating and Applying transformations ...')
    transform = transforms.Compose([
        # ImageResize(70),
        ImageResizeByCoefficient(32),
        NumpyImageToTensor(),
        FilterOutAlphaChannel(),
        ToFloatTensor()
    ])

    input = transform(thermal_roi)
    input = input.to(dtype=torch.float32, device=args.device)

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

if __name__ == "__main__":
    region_segmentation()