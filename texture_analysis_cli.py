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

from torch.utils.data import DataLoader, random_split, Subset
from torchvision.transforms import Resize, Compose, Grayscale, ToTensor, Normalize, InterpolationMode
from ignite.utils import setup_logger
from ignite.engine.events import Events
from lemanchot.core import get_profile, get_profile_names, get_or_default
from lemanchot.dataset import (
    SegmentationDataset,
    generate_weighted_sampler,
    ImageDataset,
)
from lemanchot.pipeline import load_segmentation
from lemanchot.transform import (
    BothCompose,
    BothToTensor,
    BothNormalize,
    TargetDilation,
    TrivialAugmentWide,
)

parser = argparse.ArgumentParser(description="Texture Segmentation of Inspection")
parser.add_argument(
    "--profile",
    required=True,
    choices=get_profile_names(),
    help="Select the name of profiles.",
)
parser.add_argument(
    "--mode",
    required=False,
    choices={"train", "test", "predict"},
    default="train",
    type=str,
    help="Set the mode for Dataset initialization.",
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
    # Initialize Transformation for training
    if args.mode == "train":
        input_transforms = Compose([Resize((512, 512))])
        target_transform = Compose([Resize((512, 512), InterpolationMode.NEAREST)])
        both_transforms = BothCompose(
            [
                TrivialAugmentWide(31, InterpolationMode.NEAREST),
                BothToTensor(),
                BothNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
    elif args.mode == "test":
        input_transforms = Compose([Resize((512, 512))])
        target_transform = Compose([Resize((512, 512), InterpolationMode.NEAREST)])
        both_transforms = BothCompose([
            BothToTensor(),
            BothNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # Load segmentation
    run_record = load_segmentation(
        profile_name=profile_name, database_name=dataset_name
    )
    engine = run_record["engine"]
    engine.logger = setup_logger("trainer")
    ######### Dataset & Dataloader ##########
    if args.mode == "predict":
        dataset = ImageDataset(
            root=dataset_path,
            folder_name="img",
            transforms=Compose(
                [
                    Grayscale(),
                    ToTensor(),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            ),
        )
        shuffle = False
    else:
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
        shuffle = True
        if run_record["validator"] is not None and args.mode == 'train':
            train_set_size = int(len(dataset) * 0.8)
            val_set_size = len(dataset) - train_set_size
            dataset, val_dataset = random_split(dataset, [train_set_size, val_set_size])
            val_loader = DataLoader(
                val_dataset,
                batch_size=engine.state.batch_size,
                shuffle=False,
            )
            validator = run_record["validator"]
            validator.state_dict_user_keys.append("global_epoch")
            validator.state_dict_user_keys.append("global_step")
            setattr(validator.state, "global_step", 0)
            validator.logger = setup_logger("validator")

            @engine.on(Events.EPOCH_COMPLETED(every=3))
            def run_validation(engine):
                setattr(validator.state, "global_epoch", engine.state.epoch)
                print(f"Validator global epoch = {validator.state.global_epoch}")
                validator.run(val_loader, max_epochs=1)
                vloss = get_or_default(validator.state.metrics, "loss", 0)
                print(
                    f"validation loss: {vloss:.4f}"
                )

    if profile.weight_dataset and args.mode == "train":
        # This function is very long.
        sampler = generate_weighted_sampler(dataset.gt_dataset)
        shuffle = False
    else:
        sampler = None

    data_loader = DataLoader(
        dataset,
        batch_size=engine.state.batch_size,
        shuffle=shuffle,
        sampler=sampler,
    )
    # Run the pipeline
    state = engine.run(data_loader, max_epochs=engine.state.max_epoch)
    print(state)

    return 0


if __name__ == "__main__":
    print("The experiment started ...")
    try:
        sys.exit(main())
    except Exception as ex:
        logging.exception(ex)
    print("The experiment finished ...")
