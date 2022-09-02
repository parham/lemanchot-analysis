
import sys
import logging

import torch
from torch.utils.data import DataLoader

from ignite.utils import setup_logger

from gimp_labeling_converter import XCFDataset
from lemanchot.pipeline import load_segmentation
from lemanchot.transform import GrayToRGB, ImageResizeByCoefficient, NumpyImageToTensor


def main():
    # Initialize Transformation
    transform = torch.nn.Sequential(
        GrayToRGB(),
        ImageResizeByCoefficient(32),
        NumpyImageToTensor()
    )
    dataset = XCFDataset(
        root_dir='/home/phm/Datasets/Laval_Road_9h52',
        category={
            'Metal' : 2,
            'Vegetation' : 4,
            'Pavement' : 5,
            'Wood' : 7,
            'Water' : 9
        },
        transform=transform
    )
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
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