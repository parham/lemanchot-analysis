
""" 
    @project LeManchot-Analysis : Multi-Modal Texture Analysis to Enhance Drone-based Thermographic Inspection of Structures
    @organization Laval University
    @lab MiViM Lab
    @supervisor Professor Xavier Maldague
    @industrial-partner TORNGATS
"""

import os
import random
import sys
import unittest

import torch
import numpy as np

from PIL import Image
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())
sys.path.append(__file__)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from lemanchot.dataset import Huang2020Dataset
from lemanchot.augmentation import generate_augmented_texture

class TestAugmentation(unittest.TestCase):
    
    def test_Augmentation_Texture(self):
        img = np.asarray(Image.open('/home/phm/Pictures/tex3.jpeg'), dtype=np.uint8)

        dataset = Huang2020Dataset(
            root_dir = '/home/phm/Datasets/texture/train',
            categories={
                'plaid' : 1,
                'fur' : 2,
                'brick' : 3,
                'pebbles' : 4,
                'stone' : 5,
                'wood' : 6,
                'water' : 7,
                'wall' : 8
            }
        )
        batch_size = 20
        img, target = generate_augmented_texture(
            img, dataset, batch_size, 
            random.randint(4, 30)
        )

        Image.fromarray(img).show()
        Image.fromarray(target).show()

if __name__ == '__main__':
    unittest.main()

