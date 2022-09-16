

""" 
    @title A Deep Semi-supervised Segmentation Approach for Thermographic Analysis of Industrial Components
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
# from torchvision.transforms import ToPILImage, PILToTensor

from lemanchot.core import get_config
from lemanchot.loss.core import load_loss
from lemanchot.models.core import load_model
from lemanchot.pipeline.core import load_optimizer
from lemanchot.tools.control_point import cpselect
from lemanchot.transform import FilterOutAlphaChannel, ImageResize, ImageResizeByCoefficient, NumpyImageToTensor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

known_args = []
parser = argparse.ArgumentParser(description="ROI Segmentation of Thermal Image")
parser.add_argument('tfile', type=str, help="Thermal Image file path.")
parser.add_argument('vfile', type=str, help="Visible Image file path.")
parser.add_argument('--device', type=str, default='cuda', help="The selected device.")
parser.add_argument('--iteration', type=int, default=60, help="The maximum number of iteration.")
parser.add_argument('--nclass', type=int, default=4, help="The minimum number of classes.")
parser.add_argument('--output', type=str, default='.', help="The result folder.")

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
    
    ir_filename = Path(args.tfile).stem
    ir_img = np.asarray(Image.open(args.tfile))
    if ir_img is None:
        logging.error('Failed to load the thermal image!')
        return

    vis_filename = Path(args.vfile).stem
    vis_img = np.asarray(Image.open(args.vfile))
    if vis_img is None:
        logging.error('Failed to load the visible image!')
        return

    # cps = cpselect(ir_img, vis_img)
    # source, dest = cp_to_opencv(cps)
    # homography, _ = cv2.findHomography(source, dest)
    # corrected_ir = cv2.warpPerspective(ir_img, homography, (vis_img.shape[1], vis_img.shape[0]))

    # roi = cv2.selectROI('Select Region of Interest', corrected_ir, showCrosshair=True)
    # cropped_vis = vis_img[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
    # cropped_ir = corrected_ir[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]

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
    transform = torch.nn.Sequential(
        # ImageResize(100),
        # ImageResizeByCoefficient(32),
        NumpyImageToTensor()
    )

    input = transform(ir_img)
    input = input.to(dtype=torch.float32, device=args.device)

    criterion.prepare_loss(ref=input)

    to_pil = transforms.ToPILImage()
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
    result = to_pil(result.squeeze(0).squeeze(0))

    # mat = {
    #     'homography' : homography,
    #     'visible' : np.asarray(cropped_vis),
    #     'thermal' : np.asarray(cropped_ir),
    #     'thermal_segment' : np.asarray(result)
    # }
    # savemat(os.path.join(args.output, f'{ir_filename}.mat'), mat, do_compression=True)


if __name__ == "__main__":
    region_segmentation()