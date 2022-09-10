
""" 
    @title A Deep Semi-supervised Segmentation Approach for Thermographic Analysis of Industrial Components
    @organization Laval University
    @professor  Professor Xavier Maldague
    @author     Parham Nooralishahi
    @email      parham.nooralishahi@gmail.com
"""

import numpy as np
from dotmap import DotMap
from sklearn.cluster import DBSCAN, KMeans, MeanShift, estimate_bandwidth

from lemanchot.models.core import model_register, BaseModule


@model_register('dbscan')
class DBSCANModule(BaseModule):
    def __init__(self, name : str, config : DotMap) -> None:
        super().__init__(
            name='dbscan',
            config=config
        )
        self.clss = DBSCAN(
            eps = self.eps,
            min_samples = self.min_samples,
            leaf_size = self.leaf_size
        )
    
    def forward(self, inputs):
        batch_size = inputs.shape[0]
        outputs = []
        for bindex in range(batch_size):
            input = inputs.take(indices=bindex, axis=0)
            input_size = input.shape
            db = self.clss.fit(inputs)
            output = np.uint8(db.labels_.reshape(input_size[:2]))
            outputs.append(output)
        outputs = np.dstack(outputs)
        return outputs