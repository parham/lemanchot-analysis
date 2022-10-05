
import os
import glob
import numpy as np

from PIL import Image

from torch.utils.data import Dataset

class FusionDataset(Dataset):
    def __init__(self, root_dir : str) -> None:
        super().__init__()
        # Root Directory
        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f'{root_dir} does not exist!')
        self.root_dir = root_dir
        self.thermal_dir = os.path.join(root_dir, 'thermal')
        if not os.path.join(self.thermal_dir):
            raise FileNotFoundError(f'thermal folder does not exist!')
        self.visible_dir = os.path.join(root_dir, 'visible')
        if not os.path.join(self.visible_dir):
            raise FileNotFoundError(f'visible folder does not exist!')
        # List of files
        self.coupled_files = []
        files = glob.glob(os.path.join(self.visible_dir, '*.jpg'))
        for vfile in files:
            vfname = os.path.basename(vfile) # *_VIS.jpg
            tfname = vfname.replace('VIS', 'IR')
            tfile = os.path.join(self.thermal_dir, tfname)
            if os.path.isfile(tfile):
                self.coupled_files.append((vfile, tfile))
    
    def __len__(self):
        return len(self.coupled_files)
    
    def __getitem__(self, idx):
        batch = self.coupled_files[idx]
        visible = Image.open(batch[0])
        thermal = Image.open(batch[1])
        return visible, thermal, os.path.basename(batch[0]), os.path.basename(batch[1])
