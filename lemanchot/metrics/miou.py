
"""
    @project LeManchot-Analysis : Core components
    @organization Laval University
    @lab MiViM Lab
    @supervisor Professor Xavier Maldague
    @industrial-partner TORNGATS
"""

import numpy as np

from comet_ml import Experiment

from ignite.engine import Engine
from ignite.exceptions import NotComputableError

from lemanchot.metrics.core import BaseMetric, metric_register
from lemanchot.processing import mIoU_func

@metric_register('mIoU')
class mIoU(BaseMetric):
    def __init__(self, config) -> None:
        """measure mIoU metric

        Args:
            config (_type_): _description_
            *    ignored_class (_type_): _description_
            *    iou_thresh (float, optional): _description_. Defaults to 0.1.
        """
        super().__init__(config)
        self._mIoU = 0.0
        self._mIoU_count = 0
        self._iou_map = None

    def reset(self):
        self._mIoU = 0.0
        self._mIoU_count = 0
        super(mIoU, self).reset()

    def update(self, data, **kwargs):
        output, target = data[-2], data[-1]
        iou, iou_map, maxv, maxind, _, _ = mIoU_func(output, target, iou_thresh=self.iou_thresh)
        self._mIoU += iou
        self._mIoU_count += 1
        self._iou_map = iou_map

    def compute(self,
        engine : Engine,
        experiment : Experiment,
        prefix : str = '',
        **kwargs
    ):
        if self._mIoU_count == 0:
            raise NotComputableError()

        metric = float(self._mIoU) / float(self._mIoU_count)

        experiment.log_table('iou.csv', self._iou_map)
        experiment.log_metric(
            name=f'{prefix}{self.get_name()}',
            value=metric,
            step=engine.state.iteration,
            epoch=engine.state.epoch
        )
        return metric
