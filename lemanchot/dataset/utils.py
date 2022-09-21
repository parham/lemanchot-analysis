from torch import stack as torch_stack
from torch.utils.data import Dataset, WeightedRandomSampler

def generate_weighted_sampler(
    dataset: Dataset, batch_size: int, w_type: str = "squared", replacement: bool = True
) -> WeightedRandomSampler:
    scale = 2 if w_type == "squared" else 1
    sum = 0.0

    ind_weight = list()
    for idx in range(len(dataset)):
        _, sample = dataset[idx]
        ind_weight.append(sample.sum(dim=(-2, -1)))
        sum += ind_weight[-1]

    sum = sum.reciprocal() ** scale
    ind_weight = torch_stack(ind_weight, dim=0).mul(sum)
    return WeightedRandomSampler(
        weights=ind_weight.sum(dim=1).tolist(), num_samples=batch_size, replacement=replacement
    )