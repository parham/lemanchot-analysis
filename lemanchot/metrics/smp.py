

from dotmap import DotMap
from comet_ml import Experiment

from ignite.engine import Engine

import segmentation_models_pytorch as smp

from lemanchot.metrics.core import BaseMetric, metric_register


@metric_register('smp')
class SMP_Metrics(BaseMetric):
    def __init__(self, config) -> None:
        super().__init__(config)
        # {
        #       "mode" : "multiclass", # "binary", "multilabel"
        #       "ignore_index" : 0, # Default: None
        #       "threshold" : 0.1, # Default: None
        #       "num_classes" : 6,
        #       "metrics" : {
        #           "fbeta" : {
        #               "beta" : 1.0,
        #               "reduction" : None, # For 'binary' case 'micro' = 'macro' = 'weighted' and 'micro-imagewise' = 'macro-imagewise' = 'weighted-imagewise'
        #               "class_weights" : None, # or [0.3, 0.53, 0.1, ...]
        #               "zero_division" : 1
        #           },
        #           "f1" : {
        #               "reduction" : None, # For 'binary' case 'micro' = 'macro' = 'weighted' and 'micro-imagewise' = 'macro-imagewise' = 'weighted-imagewise'
        #               "class_weights" : None, # or [0.3, 0.53, 0.1, ...]
        #               "zero_division" : 1
        #           },
        #           "iou" : {
        #               "reduction" : None, # For 'binary' case 'micro' = 'macro' = 'weighted' and 'micro-imagewise' = 'macro-imagewise' = 'weighted-imagewise'
        #               "class_weights" : None, # or [0.3, 0.53, 0.1, ...]
        #               "zero_division" : 1
        #           },
        #           "accuracy" : {
        #               "reduction" : None, # For 'binary' case 'micro' = 'macro' = 'weighted' and 'micro-imagewise' = 'macro-imagewise' = 'weighted-imagewise'
        #               "class_weights" : None, # or [0.3, 0.53, 0.1, ...]
        #               "zero_division" : 1
        #           },
        #           "precision" : {
        #               "reduction" : None, # For 'binary' case 'micro' = 'macro' = 'weighted' and 'micro-imagewise' = 'macro-imagewise' = 'weighted-imagewise'
        #               "class_weights" : None, # or [0.3, 0.53, 0.1, ...]
        #               "zero_division" : 1
        #           },
        #           "recall" : {
        #               "reduction" : None, # For 'binary' case 'micro' = 'macro' = 'weighted' and 'micro-imagewise' = 'macro-imagewise' = 'weighted-imagewise'
        #               "class_weights" : None, # or [0.3, 0.53, 0.1, ...]
        #               "zero_division" : 1
        #           },
        #           "sensitivity" : {
        #               "reduction" : None, # For 'binary' case 'micro' = 'macro' = 'weighted' and 'micro-imagewise' = 'macro-imagewise' = 'weighted-imagewise'
        #               "class_weights" : None, # or [0.3, 0.53, 0.1, ...]
        #               "zero_division" : 1
        #           },
        #           "specificity" : {
        #               "reduction" : None, # For 'binary' case 'micro' = 'macro' = 'weighted' and 'micro-imagewise' = 'macro-imagewise' = 'weighted-imagewise'
        #               "class_weights" : None, # or [0.3, 0.53, 0.1, ...]
        #               "zero_division" : 1
        #           },
        #           "balanced_accuracy" : {
        #               "reduction" : None, # For 'binary' case 'micro' = 'macro' = 'weighted' and 'micro-imagewise' = 'macro-imagewise' = 'weighted-imagewise'
        #               "class_weights" : None, # or [0.3, 0.53, 0.1, ...]
        #               "zero_division" : 1
        #           },
        #           "ppv" : {
        #               "reduction" : None, # For 'binary' case 'micro' = 'macro' = 'weighted' and 'micro-imagewise' = 'macro-imagewise' = 'weighted-imagewise'
        #               "class_weights" : None, # or [0.3, 0.53, 0.1, ...]
        #               "zero_division" : 1
        #           },
        #           "npv" : {
        #               "reduction" : None, # For 'binary' case 'micro' = 'macro' = 'weighted' and 'micro-imagewise' = 'macro-imagewise' = 'weighted-imagewise'
        #               "class_weights" : None, # or [0.3, 0.53, 0.1, ...]
        #               "zero_division" : 1
        #           },
        #           "fnr" : {
        #               "reduction" : None, # For 'binary' case 'micro' = 'macro' = 'weighted' and 'micro-imagewise' = 'macro-imagewise' = 'weighted-imagewise'
        #               "class_weights" : None, # or [0.3, 0.53, 0.1, ...]
        #               "zero_division" : 1
        #           },
        #           "fpr" : {
        #               "reduction" : None, # For 'binary' case 'micro' = 'macro' = 'weighted' and 'micro-imagewise' = 'macro-imagewise' = 'weighted-imagewise'
        #               "class_weights" : None, # or [0.3, 0.53, 0.1, ...]
        #               "zero_division" : 1
        #           },
        #           "fdr" : {
        #               "reduction" : None, # For 'binary' case 'micro' = 'macro' = 'weighted' and 'micro-imagewise' = 'macro-imagewise' = 'weighted-imagewise'
        #               "class_weights" : None, # or [0.3, 0.53, 0.1, ...]
        #               "zero_division" : 1
        #           },
        #           "for" : {
        #               "reduction" : None, # For 'binary' case 'micro' = 'macro' = 'weighted' and 'micro-imagewise' = 'macro-imagewise' = 'weighted-imagewise'
        #               "class_weights" : None, # or [0.3, 0.53, 0.1, ...]
        #               "zero_division" : 1
        #           },
        #           "lr+" : {
        #               "reduction" : None, # For 'binary' case 'micro' = 'macro' = 'weighted' and 'micro-imagewise' = 'macro-imagewise' = 'weighted-imagewise'
        #               "class_weights" : None, # or [0.3, 0.53, 0.1, ...]
        #               "zero_division" : 1
        #           },
        #           "lr-" : {
        #               "reduction" : None, # For 'binary' case 'micro' = 'macro' = 'weighted' and 'micro-imagewise' = 'macro-imagewise' = 'weighted-imagewise'
        #               "class_weights" : None, # or [0.3, 0.53, 0.1, ...]
        #               "zero_division" : 1
        #           }
        #       }
        # }

    def update(self, batch, **kwargs):
        output, target = batch[-2], batch[-1]
        tp, fp, fn, tn = smp.metrics.get_stats(output, target, 
            mode=self.mode, 
            threshold=self.threshold, 
            ignore_index=
        )

    def compute(self,
        engine : Engine,
        experiment : Experiment,
        prefix : str = '',
        **kwargs
    ):
        pass