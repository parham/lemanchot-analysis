
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
from pathlib import Path
import sys

from tqdm import tqdm

from torchvision.transforms import Compose
from torch.utils.data import DataLoader

from PIL import Image


sys.path.append(os.getcwd())
sys.path.append(__file__)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from lemanchot.core import get_config, get_device, get_experiment, get_profile, get_profile_names
from lemanchot.loss.core import load_loss
from lemanchot.models.core import load_model
from lemanchot.pipeline.core import load_optimizer
from lemanchot.pipeline.saver import ImageSaver
from lemanchot.dataset.mat import MATLABDataset
from lemanchot.methods import iterative_region_segmentation
from lemanchot.transform import (
    FilterOutAlphaChannel,
    ImageResize,
    ImageResizeByCoefficient,
    NumpyImageToTensor,
    ToFloatTensor,
    ToUINT8Tensor
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

parser = argparse.ArgumentParser(description="Multi-Modal Analysis")
parser.add_argument('dir', type=str, help="Multi-Modal file")
parser.add_argument('--out', type=str, help="The output directory")
parser.add_argument('--profile', required=True, choices=get_profile_names(), help="Select the name of profiles.")
parser.add_argument('--iteration', type=int, default=60, help="The maximum number of iteration.")
parser.add_argument('--nclass', type=int, default=10, help="The minimum number of classes.")

def main():
    args = parser.parse_intermixed_args()
    parser.print_help()

    if not os.path.isdir(args.dir):
        logging.error(f'{args.dir} does not exist!')
        return
    
    Path(args.out).mkdir(parents=True, exist_ok=True)

    # Determine the selected device to use for processing
    device = get_device()
    profile = get_profile(args.profile)
    # Load the experiment configuration
    experiment_name = get_profile(args.profile).experiment_config_name
    experiment_config = get_config(experiment_name)
    # Create the experiment
    experiment = get_experiment(args.profile, 'piping_inspection')
    # Create transformations
    logging.info('Creating and Applying transformations ...')
    transforms = Compose([
        ImageResize(100),
        ImageResizeByCoefficient(32),
        NumpyImageToTensor(),
        FilterOutAlphaChannel(),
        ToUINT8Tensor()
    ])
    # Create the dataset
    dataset = MATLABDataset(
        root_dir = args.dir,
        input_tag = 'ir_roi',
        transforms = transforms
    )
    data_loader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=True
    )
    logging.info('Dataset is created ...')
    ################ Initialization ################
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

    # Create the image saver
    logging.info('Creating and Initialize Image Saver ...')
    img_saver = ImageSaver(args.out)
    ################ Run the Experiment ################
    for data in iter(data_loader):
        fpath = data[-1][0]
        fname = Path(fpath).stem
        logging.info(f'Processing ... {fpath}')
        process_obj = iterative_region_segmentation(
            batch = data,
            experiment_config = experiment_config,
            device = device,
            num_iteration = args.iteration,
            class_count_limit = args.nclass,
            model = model, 
            criterion = criterion, 
            optimizer = optimizer
        )
        res = None
        step = 1
        for out in tqdm(process_obj):
            output = out['y_pred']
            output = output.squeeze(0).squeeze(0).cpu().detach().numpy()
            # Logging the loss metric
            experiment.log_metrics({
                'loss' : out['loss'],
                'class_count' : out['class_count']
            }, step=step, epoch=1)
            res = output
            step += 1
        # Logging the output image into comet.ml
        experiment.log_image(output, name=fname, step=step)
        # Logging the image
        img_saver(fname, res)
        logging.info(f'Saving the output image ... {fname}')

if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        logging.exception(ex)