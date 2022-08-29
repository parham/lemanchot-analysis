
from torch.optim import Optimizer
import torch.optim as optim

from dotmap import DotMap

from lemanchot.models.core import BaseModule

def load_optimizer(model : BaseModule, experiment_config : DotMap) -> Optimizer:
    params = model.parameters()
    optim_name = experiment_config.name
    optim_config = experiment_config.config
    return {
        'SGD' : lambda ps, config: optim.SGD(ps, **config),
        'Adam' : lambda ps, config: optim.Adam(ps, **config)
    }[optim_name](params, optim_config)
