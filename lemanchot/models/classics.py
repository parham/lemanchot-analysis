
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
        self.clss = DBSCAN(**config)
    
    def forward(self, input):
        input_size = input.shape
        data = np.float32(input.reshape((-1, 3)))
        db = self.clss.fit(data)
        output = np.uint8(db.labels_.reshape(input_size[:2]))
        return output