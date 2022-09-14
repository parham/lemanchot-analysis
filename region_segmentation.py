

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

from PIL import Image
from pathlib import Path
from tqdm import trange

import torch
import torchvision.transforms as transforms
# from torchvision.transforms import ToPILImage, PILToTensor

from lemanchot.core import get_config
from lemanchot.loss.core import load_loss
from lemanchot.models.core import load_model
from lemanchot.pipeline.core import load_optimizer
from lemanchot.transform import FilterOutAlphaChannel, ImageResize, ImageResizeByCoefficient, NumpyImageToTensor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

known_args = []
parser = argparse.ArgumentParser(description="ROI Segmentation of Thermal Image")
parser.add_argument('file', type=str, help="Image file path.")
parser.add_argument('--device', type=str, default='cuda', help="The selected device.")
parser.add_argument('--iteration', type=int, default=60, help="The maximum number of iteration.")
parser.add_argument('--nclass', type=int, default=4, help="The minimum number of classes.")
parser.add_argument('--output', type=str, default='.', help="The result folder.")

def region_segmentation():
    args = parser.parse_intermixed_args()
    parser.print_help()

    if not os.path.isfile(args.file):
        logging.error(f'{args.file} does not exist!')
        return
    
    filename = Path('/root/dir/sub/file.ext').stem
    img = Image.open(args.file)
    if img is None:
        logging.error('Failed to load the image!')
        return
    
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
        ImageResize(100),
        ImageResizeByCoefficient(32),
        NumpyImageToTensor()
    )

    input = transform(img)
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
    result.save(os.path.join(args.output, f'{filename}.jpg'))
    

if __name__ == "__main__":
    region_segmentation()