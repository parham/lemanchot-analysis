
import argparse
import sys
import logging

import torch
from torch.utils.data import DataLoader

from ignite.utils import setup_logger

from gimp_labeling_converter import XCFDataset
from lemanchot.core import get_profile, get_profile_names
from lemanchot.pipeline import load_segmentation
from lemanchot.transform import FilterOutAlphaChannel, GrayToRGB, ImageResizeByCoefficient, NumpyImageToTensor

parser = argparse.ArgumentParser(description="Texture Segmentation of Inspection")
parser.add_argument('--profile', required=True, choices=get_profile_names(), help="Select the name of profiles.")
# parser.add_argument('--profile', '-p', type=str, required=True, help="The name of the profile")

def main():
    args = parser.parse_args()
    parser.print_help()
    profile_name = args.profile
    # Load Settings
    profile = get_profile(profile_name)
    dataset_path = profile.dataset_path
    categories = profile.categories
    # Initialize Transformation
    transform = torch.nn.Sequential(
        GrayToRGB(),
        ImageResizeByCoefficient(32),
        NumpyImageToTensor(),
        FilterOutAlphaChannel()
    )
    dataset = XCFDataset(
        root_dir=dataset_path,
        category=categories,
        transform=transform
    )
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    # Load segmentation
    run_record = load_segmentation(profile_name='parham', database_name='Laval_Road_9h52')
    engine = run_record['engine']
    engine.logger = setup_logger('trainer')
    # Run the pipeline
    state = engine.run(data_loader, max_epochs=engine.state.max_epoch)
    print(state)
    return 0

if __name__ == '__main__':
    print('The experiment is started ...')
    try:
        sys.exit(main())
    except Exception as ex:
        logging.exception(ex)
    print('The experiment is finished ...')