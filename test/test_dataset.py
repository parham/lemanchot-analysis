

import os
import sys
import unittest
import torch
import numpy as np

from torch.utils.data import DataLoader

sys.path.append(os.getcwd())
sys.path.append(__file__)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from lemanchot.dataset import Huang2020Dataset

class TestDataset(unittest.TestCase):
    
    def test_Huang2020Dataset(self):
        dataset = Huang2020Dataset(
            root_dir = '/home/phm/Datasets/texture/train',
            categories={
                'linen' : 1,
                'styrofoam' : 2,
                'ceiling' : 3,
                'sand' : 4,
                'stone' : 5,
                'wood' : 6,
                'water' : 7,
                'wall' : 8
            }
        )
        batch_size = 2
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for img, target in data_loader:
            assert len(torch.unique(target)) <= batch_size

if __name__ == '__main__':
    unittest.main()