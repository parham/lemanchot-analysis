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

from torch.cuda import device_count
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
from ignite.utils import setup_logger

from lemanchot.core import get_profile, get_profile_names
from lemanchot.dataset import SegmentationDataset, generate_weighted_sampler
from lemanchot.pipeline import load_segmentation
from lemanchot.transform import (
    BothCompose,
    BothToTensor,
    TrivialAugmentWide,
    InterpolationMode,
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
    target_transform = Resize((512, 512), InterpolationMode.NEAREST)
    both_transforms = BothCompose(
        [TrivialAugmentWide(31, InterpolationMode.NEAREST), BothToTensor()]
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
    if profile.weight_dataset:
        # This function is very long.
        sampler = generate_weighted_sampler(dataset.gt_dataset)
    else:
        sampler = None

    data_loader = DataLoader(
        dataset,
        batch_size=engine.state.batch_size,
        shuffle=True if sampler is None else None,
        sampler=sampler,
    )

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
