
"""
    @project LeManchot-Analysis : Core components
    @organization Laval University
    @lab MiViM Lab
    @supervisor Professor Xavier Maldague
    @industrial-partner TORNGATS
"""

from typing import Dict

from gimp_labeling_converter import XCFDataset

class RepetitiveDatasetWrapper(XCFDataset):
    def __init__(
        self,
        root_dir: str,
        category: Dict[str, int],
        transform=None,
        target_transform=None,
    ) -> None:
        super().__init__(root_dir, category, transform, target_transform)

    @property
    def wrapped_dataset(self):
        return self.dataset_

    def __len__(self):
        return super().__len__() * self.iteration

    def __getitem__(self, idx):
        super().__getitem__(idx % self.actual_size)