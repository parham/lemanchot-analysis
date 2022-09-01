
from torch.utils.data import DataLoader

from ignite.utils import setup_logger

from gimp_labeling_converter import XCFDataset
from lemanchot.pipeline import load_segmentation


def main():
    dataset = XCFDataset(
        root_dir='/home/phm/Datasets/Laval_Road_9h52',
        category={
            'Metal' : 2,
            'Vegetation' : 4,
            'Pavement' : 5,
            'Wood' : 7,
            'Water' : 9
        }
    )
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    # Load segmentation
    engine = load_segmentation()
    engine.logger = setup_logger('trainer')

if __name__ == '__main__':
    print('Test Segmentation 01 : ')
    main()