

""" 
    @title A Deep Semi-supervised Segmentation Approach for Thermographic Analysis of Industrial Components
    @organization Laval University
    @professor  Professor Xavier Maldague
    @author     Parham Nooralishahi
    @email      parham.nooralishahi@gmail.com
"""

import argparse
import os
import sys
import logging
import warnings
import numpy as np

from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageOps, ImageFilter

from torch.utils.data import DataLoader
from torchvision.transforms import Compose

sys.path.append(os.getcwd())
sys.path.append(__file__)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from use_cases.use_case_01.vggfusion import fuse
from use_cases.use_case_01.dataset import FusionDataset

warnings.simplefilter('ignore')
logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

parser = argparse.ArgumentParser(description="Fusion of thermal and visible images to enahnce detection.")
parser.add_argument('input', type=str, help="input directory.")
parser.add_argument('--output', required=False, type=str, help="output directory.")

def main():
    args = parser.parse_args()
    parser.print_help()
    # Create Dataset
    logging.info('Create dataset!')
    fusdataset = FusionDataset(args.input)
    # data_loader = DataLoader(
    #     fusdataset, 
    #     batch_size=1, 
    #     shuffle=True
    # )
    # Create the result folder
    fuse_dir = os.path.join(args.input, 'fused') if not args.output else args.output
    Path(fuse_dir).mkdir(parents=True, exist_ok=True)
    # Create transformations
    logging.info('Creating and Applying transformations ...')

    pbar = tqdm(range(len(fusdataset)))
    for index in pbar:
        batch = fusdataset[index]
        pbar.set_description(f'Processing {batch[-2]}')
        # Preprocessing
        visimg = ImageOps.grayscale(batch[0])
        thimg = ImageOps.grayscale(batch[1])
        thimg = ImageOps.invert(thimg)
        thimg = thimg.filter(ImageFilter.EDGE_ENHANCE_MORE)
        viz = np.asarray(visimg)
        ir = np.asarray(thimg)
        # Fusion of thermal and visible images
        res = fuse(viz, ir)
        # Save the images
        fusefile = os.path.join(fuse_dir, batch[-2].replace('IR', 'FUSED'))
        Image.fromarray((res * 255).astype(np.uint8)).save(fusefile)

if __name__ == '__main__':
    try:
        logging.info('Experiment is started ... ')
        main()
        logging.info('Experiment is finished.')
    except Exception as ex:
        logging.error('Experiment is failed!')
        logging.exception(ex)
    