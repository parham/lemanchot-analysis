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

from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose,
    RandomRotation,
    Resize,
    Grayscale,
    ColorJitter,
    ToTensor,
)
from ignite.utils import setup_logger

from gimp_labeling_converter import XCFDataset
from lemanchot.core import get_profile, get_profile_names
from lemanchot.dataset import SegmentationDataset
from lemanchot.pipeline import load_segmentation
from lemanchot.transform import (
    BothRandomRotate,
    FilterOutAlphaChannel,
    ImageResize,
    ImageResizeByCoefficient,
    NumpyImageToTensor,
)

parser = argparse.ArgumentParser(description="Texture Segmentation of Inspection")
parser.add_argument(
    "--profile",
    required=True,
    choices=get_profile_names(),
    help="Select the name of profiles.",
)
# parser.add_argument('--profile', '-p', type=str, required=True, help="The name of the profile")


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
    input_transforms = Compose([Grayscale(), Resize((512, 512)), ToTensor()])
    target_transform = Compose([Resize((512, 512))])
    both_transforms = BothRandomRotate(angles=(0, 15, 30, 45, 60, 75, 90))
    # transform = torch.nn.Sequential(
    # ImageResize(70),
    # ImageResizeByCoefficient(32),
    # NumpyImageToTensor(),
    # FilterOutAlphaChannel(),
    # )
    # target_transform = torch.nn.Sequential(
    # ImageResize(70),
    # ImageResizeByCoefficient(32),
    # NumpyImageToTensor(),
    # FilterOutAlphaChannel(),
    # )
    # Load segmentation
    run_record = load_segmentation(
        profile_name=profile_name, database_name=dataset_name
    )
    engine = run_record["engine"]
    engine.logger = setup_logger("trainer")
    ######### Dataset & Dataloader ##########
    # dataset = XCFDataset(
    # root_dir=dataset_path,
    # category=categories,
    # transform=transform,
    # target_transform=target_transform,
    # )
    dataset = SegmentationDataset(
        root=dataset_path,
        img_folder = "img",
        img_ext= ".jpg",
        gt_folder = "gt",
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
