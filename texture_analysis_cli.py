"""
    @project LeManchot-Analysis : Multi-Modal Texture Analysis to Enhance Drone-based Thermographic Inspection of Structures
    @organization Laval University
    @lab MiViM Lab
    @supervisor Professor Xavier Maldague
    @industrial-partner TORNGATS
"""

import argparse
import sys
import logging

from torch.nn import Sequential
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
from ignite.utils import setup_logger

from lemanchot.core import get_profile, get_profile_names
from lemanchot.dataset import SegmentationDataset
from lemanchot.pipeline import load_segmentation
from lemanchot.transform import (
    BothToTensor,
    TrivialAugmentWide,
)

parser = argparse.ArgumentParser(description="Texture Segmentation of Inspection")
parser.add_argument(
    "--profile",
    required=True,
    choices=get_profile_names(),
    help="Select the name of profiles.",
)


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
    input_transforms = Resize((512, 512))
    target_transform = Resize((512, 512))
    both_transforms = Sequential(
        TrivialAugmentWide(31, "bilinear"),
        BothToTensor(),
    )
    # Load segmentation
    run_record = load_segmentation(
        profile_name=profile_name, database_name=dataset_name
    )
    engine = run_record["engine"]
    engine.logger = setup_logger("trainer")
    ######### Dataset & Dataloader ##########
    dataset = SegmentationDataset(
        root=dataset_path,
        img_folder="img",
        img_ext=".png",
        gt_folder="gt",
        classes=categories,
        input_transforms=input_transforms,
        target_transforms=target_transform,
        both_transforms=both_transforms,
    )
    data_loader = DataLoader(dataset, batch_size=engine.state.batch_size, shuffle=True)

    # Run the pipeline
    state = engine.run(data_loader, max_epochs=engine.state.max_epoch)
    print(state)

    return 0


if __name__ == "__main__":
    print("The experiment is started ...")
    try:
        sys.exit(main())
    except Exception as ex:
        logging.exception(ex)
    print("The experiment is finished ...")
