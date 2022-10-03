
""" 
    @project LeManchot-Analysis : Multi-Modal Texture Analysis to Enhance Drone-based Thermographic Inspection of Structures
    @organization Laval University
    @lab MiViM Lab
    @supervisor Professor Xavier Maldague
    @industrial-partner TORNGATS
"""

from torch import stack as torch_stack
from torch.utils.data import Dataset, WeightedRandomSampler

def generate_weighted_sampler(
    dataset: Dataset, w_type: str = "squared", replacement: bool = True
) -> WeightedRandomSampler:
    """
    Function to auto generate a weighted random sampler for a 
    specific dataset. Dataset statistics are computed at runtime 
    which can be very long procedure.

    Args:
        dataset (Dataset): Dataset object to balance
        w_type (str, optional): Type of weighting to apply to each sample. 
                                Defaults to "squared". Any other option will be
                                a linear weighting.
        replacement (bool, optional): if ``True``, samples are drawn with replacement.
        If not, they are drawn without replacement, which means that when a sample index 
        is drawn for a row, it cannot be drawn again for that row. Defaults to True.

    Returns:
        WeightedRandomSampler
    """
    scale = 2 if w_type == "squared" else 1
    sum = 0.0
    ind_weight = list()
    for idx in range(len(dataset)):
        sample = dataset[idx]
        ind_weight.append(sample.sum(dim=(-2, -1)))
        sum += ind_weight[-1]

    sum = sum.reciprocal() ** scale
    ind_weight = torch_stack(ind_weight, dim=0).mul(sum)
    return WeightedRandomSampler(
        weights=ind_weight.sum(dim=1).tolist(), num_samples=len(dataset), replacement=replacement
    )