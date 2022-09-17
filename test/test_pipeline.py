"""
    @project LeManchot-Analysis : Multi-Modal Texture Analysis to Enhance Drone-based Thermographic Inspection of Structures
    @organization Laval University
    @lab MiViM Lab
    @supervisor Professor Xavier Maldague
    @industrial-partner TORNGATS
"""

import argparse
import os
import sys
import logging
import unittest

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
from ignite.metrics import SSIM, Bleu, mIoU, ConfusionMatrix
from ignite.metrics.metric import BatchWise

from gimp_labeling_converter import XCFDataset

sys.path.append(os.getcwd())
sys.path.append(__file__)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from lemanchot.core import get_profile, get_profile_names
from lemanchot.dataset import SegmentationDataset
from lemanchot.pipeline import load_segmentation
from lemanchot.transform import (
    BothRandomRotate,
    FilterOutAlphaChannel,
    ImageResize,
    ImageResizeByCoefficient,
    NumpyImageToTensor,
    ToGrayscale,
)

class TestDataset(unittest.TestCase):
    
    def test_pipeline(self):

        profile_name = 'zephyrus-ulaval'

        ######### Settings ##########
        profile = get_profile(profile_name)
        dataset_name = profile.dataset_name
        dataset_path = profile.dataset_path
        categories = profile.categories
        ######### Transformation ##########
        # Initialize Transformation
        transform = Compose([
            ImageResize(70),
            ImageResizeByCoefficient(32),
            NumpyImageToTensor(),
            FilterOutAlphaChannel()
        ])
        target_transform = Compose([
            ImageResize(70),
            ImageResizeByCoefficient(32),
            NumpyImageToTensor(),
            FilterOutAlphaChannel()
        ])
        # Load segmentation
        run_record = load_segmentation(
            profile_name=profile_name, database_name=dataset_name
        )
        engine = run_record["engine"]
        engine.logger = setup_logger("trainer")

        # metric = SSIM(data_range=1.0)
        # metric.attach(engine, "ssim_ignite", usage=BatchWise())

        # metric = Bleu(ngram=4, smooth="smooth1")
        # metric.attach(engine, "bleu_ignite", usage=BatchWise())

        ######### Dataset & Dataloader ##########
        dataset = XCFDataset(
            root_dir=dataset_path,
            category=categories,
            transform=transform,
            target_transform=target_transform,
        )
        data_loader = DataLoader(dataset, 
            batch_size=engine.state.batch_size, 
            shuffle=True
        )

        # Run the pipeline
        state = engine.run(data_loader, max_epochs=engine.state.max_epoch)
        print(state)

if __name__ == '__main__':
    unittest.main()