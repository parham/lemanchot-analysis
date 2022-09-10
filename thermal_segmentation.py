
""" 
    @title A Deep Semi-supervised Segmentation Approach for Thermographic Analysis of Industrial Components
    @organization Laval University
    @professor  Professor Xavier Maldague
    @author     Parham Nooralishahi
    @email      parham.nooralishahi@gmail.com
"""

import argparse
import sys
import logging

import torch
from torch.utils.data import DataLoader

from ignite.utils import setup_logger

from gimp_labeling_converter import XCFDataset
from lemanchot.core import get_profile, get_profile_names
from lemanchot.pipeline import load_segmentation
from lemanchot.transform import FilterOutAlphaChannel, ImageResize, ImageResizeByCoefficient, NumpyImageToTensor

parser = argparse.ArgumentParser(description="A Deep Semi-supervised Segmentation Approach for Thermographic Analysis of Industrial Components")
parser.add_argument('--profile', required=True, choices=get_profile_names(), help="Select the name of profiles.")

def main():
    args = parser.parse_args()
    parser.print_help()
    profile_name = args.profile
    ######### Settings ##########
    profile = get_profile(profile_name)
    dataset_name = profile.dataset_name
    dataset_path = profile.dataset_path
    categories = profile.categories
    ######### Transformation ##########
    # Initialize Transformation
    transform = torch.nn.Sequential(
        ImageResize(100),
        ImageResizeByCoefficient(32),
        NumpyImageToTensor(),
        FilterOutAlphaChannel()
    )
    target_transform = torch.nn.Sequential(
        ImageResize(100),
        ImageResizeByCoefficient(32),
        NumpyImageToTensor(),
        FilterOutAlphaChannel()
    )
    # Load segmentation
    run_record = load_segmentation(
        profile_name=profile_name, 
        database_name=dataset_name
    )
    engine = run_record['engine']
    engine.logger = setup_logger('trainer')
    ######### Dataset & Dataloader ##########
    dataset = XCFDataset(
        root_dir=dataset_path,
        category=categories,
        transform=transform,
        target_transform=target_transform
    )
    data_loader = DataLoader(
        dataset, 
        batch_size=engine.state.batch_size, 
        shuffle=True
    )

    # Run the pipeline
    state = engine.run(
        data_loader, 
        max_epochs=engine.state.max_epoch
    )
    print(state)

    return 0

if __name__ == '__main__':
    print('The experiment is started ...')
    try:
        sys.exit(main())
    except Exception as ex:
        logging.exception(ex)
    print('The experiment is finished ...')