
""" 
    @title A Deep Semi-supervised Segmentation Approach for Thermographic Analysis of Industrial Components
    @organization Laval University
    @professor  Professor Xavier Maldague
    @author     Parham Nooralishahi
    @email      parham.nooralishahi@gmail.com
"""

import argparse
import logging
import os
import sys

from tqdm import trange

import torchvision.transforms as transforms

sys.path.append(os.getcwd())
sys.path.append(__file__)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

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

parser = argparse.ArgumentParser(description="Multi-Modal Analysis")
parser.add_argument('file', type=str, help="Multi-Modal file")
parser.add_argument('--iteration', type=int, default=60,
                    help="The maximum number of iteration.")
parser.add_argument('--nclass', type=int, default=10,
                    help="The minimum number of classes.")

def main():
    args = parser.parse_intermixed_args()
    parser.print_help()

    if not os.path.isfile(args.file):
        logging.error(f'{args.file} does not exist!')
        return
    
    # Determine the selected device to use for processing
    device = get_device()
    # Load the experiment configuration
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

    # Create the dataset
    

    for iter in trange(args.iteration):



if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        logging.exception(ex)