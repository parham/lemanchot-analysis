
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
from skimage import segmentation, color
from skimage.future import graph

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

@model_register('kmeans')
class KMeanModule(BaseModule):
    def __init__(self, name : str, config : DotMap) -> None:
        """KMean clustering
        Args:
            name (str): _description_
            config (DotMap): _description_
        """
        super().__init__(
            name='kmeans',
            config=config
        )
        self.clss = KMeans(**config)
    
    def forward(self, input):
        input_size = input.shape
        data = np.float32(input.reshape((-1, 3)))
        db = self.clss.fit(data)
        output = np.uint8(db.labels_.reshape(input_size[:2]))
        return output

@model_register('meanshift')
class MeanShiftModule(BaseModule):
    def __init__(self, name : str, config : DotMap) -> None:
        super().__init__(
            name='meanshift',
            config=config
        )
        self._config = config
    
    def forward(self, input):
        input_size = input.shape
        data = np.float32(input.reshape((-1, 3)))
        bandwidth = estimate_bandwidth(data, 
            quantile=self._config.quantile, 
            n_samples=self._config.n_samples)
        db = MeanShift(
            bandwidth=bandwidth, 
            bin_seeding=True
        ).fit(data)
        output = np.uint8(db.labels_.reshape(input_size[:2]))
        return output

@model_register('graphcut')
class GraphCutModule(BaseModule):
    def __init__(self, name : str, config : DotMap) -> None:
        """graphcut clustering
        Args:
            name (str): _description_
            config (DotMap): _description_
        """
        super().__init__(
            name='graphcut',
            config=config
        )
        self._config = config
    
    def forward(self, input):
        labels = segmentation.slic(input, 
            compactness=self._config.compactness, 
            n_segments=self._config.n_segments,
            start_label=1)
        g = graph.rag_mean_color(input, labels, mode='similarity')
        labels = graph.cut_normalized(labels, g)
        output = color.label2rgb(labels, input, kind='avg', bg_label=0)
        return output