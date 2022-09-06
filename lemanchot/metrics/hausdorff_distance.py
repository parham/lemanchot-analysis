
""" 
    @project LeManchot : Multi-Modal Data Acquisition and Processing of Drone-based Inspection
    @organization Laval University
    @lab MiViM Lab
    @supervisor Professor Xavier Maldague
    @industrial-partner TORNGATS
"""

import numpy as np

from scipy.spatial.distance import directed_hausdorff

from lemanchot.metrics.core import Function_Metric, metric_register 

def directed_hausdorff_distance(img: np.ndarray, target: np.ndarray, **kwargs):
    hdvalue = max(directed_hausdorff(img, target)[0], directed_hausdorff(target, img)[0])
    return {'directed_hausdorff' : hdvalue}

@metric_register('hausdorff_distance')
class Hausdorff_Distance(Function_Metric):
    def __init__(self, config):
        super().__init__(
            func=directed_hausdorff_distance,
            config=config
        )