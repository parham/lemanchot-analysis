

from dotmap import DotMap
from comet_ml import Experiment

import torch
from ignite.engine import Engine

import segmentation_models_pytorch as smp

from lemanchot.metrics.core import BaseMetric, metric_register


@metric_register('smp')
class SMP_Metrics(BaseMetric):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.metrics_stats = {}
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
        target = target.to(dtype=output.dtype)
        # Calculate the stats
        tp, fp, fn, tn = smp.metrics.get_stats(output, target, 
            mode=self.mode, 
            threshold=self.threshold if hasattr(self, 'threshold') and self.mode != 'multiclass' else None, 
            ignore_index=self.ignore_index if hasattr(self, 'ignore_index') else None,
            num_classes=self.num_classes
        )
        if not hasattr(self, 'metrics'):
            raise ValueError('No metris is defined!')
        
        self.metrics_stats = {}
        if 'fbeta' in self.metrics:
            metrics = self.metrics['fbeta']
            fbeta = smp.metrics.fbeta_score(tp,fp,fn,tn,
                beta=metrics['beta'] if 'beta' in metrics else 1.0,
                reduction=metrics['reduction'] if 'reduction' in metrics else 'micro',
                class_weights=metrics['class_weights'] if 'class_weights' in metrics else None,
                zero_division=metrics['zero_division'] if 'zero_division' in metrics else 1.0
            )
            self.metrics_stats['fbeta'] = fbeta
        if 'f1' in self.metrics:
            metrics = self.metrics['f1']
            f1 = smp.metrics.f1_score(tp,fp,fn,tn,
                reduction=metrics['reduction'] if 'reduction' in metrics else 'micro',
                class_weights=metrics['class_weights'] if 'class_weights' in metrics else None,
                zero_division=metrics['zero_division'] if 'zero_division' in metrics else 1.0
            )
            self.metrics_stats['f1'] = f1
        if 'iou' in self.metrics:
            metrics = self.metrics['iou']
            iou = smp.metrics.iou_score(tp,fp,fn,tn,
                reduction=metrics['reduction'] if 'reduction' in metrics else 'micro',
                class_weights=metrics['class_weights'] if 'class_weights' in metrics else None,
                zero_division=metrics['zero_division'] if 'zero_division' in metrics else 1.0
            )
            self.metrics_stats['iou'] = iou
        if 'accuracy' in self.metrics:
            metrics = self.metrics['accuracy']
            accuracy = smp.metrics.accuracy(tp,fp,fn,tn,
                reduction=metrics['reduction'] if 'reduction' in metrics else 'micro',
                class_weights=metrics['class_weights'] if 'class_weights' in metrics else None,
                zero_division=metrics['zero_division'] if 'zero_division' in metrics else 1.0
            )
            self.metrics_stats['accuracy'] = accuracy
        if 'precision' in self.metrics:
            metrics = self.metrics['precision']
            precision = smp.metrics.precision(tp,fp,fn,tn,
                reduction=metrics['reduction'] if 'reduction' in metrics else 'micro',
                class_weights=metrics['class_weights'] if 'class_weights' in metrics else None,
                zero_division=metrics['zero_division'] if 'zero_division' in metrics else 1.0
            )
            self.metrics_stats['precision'] = precision
        if 'recall' in self.metrics:
            metrics = self.metrics['recall']
            recall = smp.metrics.recall(tp,fp,fn,tn,
                reduction=metrics['reduction'] if 'reduction' in metrics else 'micro',
                class_weights=metrics['class_weights'] if 'class_weights' in metrics else None,
                zero_division=metrics['zero_division'] if 'zero_division' in metrics else 1.0
            )
            self.metrics_stats['recall'] = recall
        if 'sensitivity' in self.metrics:
            metrics = self.metrics['sensitivity']
            sensitivity = smp.metrics.sensitivity(tp,fp,fn,tn,
                reduction=metrics['reduction'] if 'reduction' in metrics else 'micro',
                class_weights=metrics['class_weights'] if 'class_weights' in metrics else None,
                zero_division=metrics['zero_division'] if 'zero_division' in metrics else 1.0
            )
            self.metrics_stats['sensitivity'] = sensitivity
        if 'specificity' in self.metrics:
            metrics = self.metrics['specificity']
            specificity = smp.metrics.specificity(tp,fp,fn,tn,
                reduction=metrics['reduction'] if 'reduction' in metrics else 'micro',
                class_weights=metrics['class_weights'] if 'class_weights' in metrics else None,
                zero_division=metrics['zero_division'] if 'zero_division' in metrics else 1.0
            )
            self.metrics_stats['specificity'] = specificity
        if 'balanced_accuracy' in self.metrics:
            metrics = self.metrics['balanced_accuracy']
            balanced_accuracy = smp.metrics.balanced_accuracy(tp,fp,fn,tn,
                reduction=metrics['reduction'] if 'reduction' in metrics else 'micro',
                class_weights=metrics['class_weights'] if 'class_weights' in metrics else None,
                zero_division=metrics['zero_division'] if 'zero_division' in metrics else 1.0
            )
            self.metrics_stats['balanced_accuracy'] = balanced_accuracy
        if 'ppv' in self.metrics:
            metrics = self.metrics['ppv']
            ppv = smp.metrics.positive_predictive_value(tp,fp,fn,tn,
                reduction=metrics['reduction'] if 'reduction' in metrics else 'micro',
                class_weights=metrics['class_weights'] if 'class_weights' in metrics else None,
                zero_division=metrics['zero_division'] if 'zero_division' in metrics else 1.0
            )
            self.metrics_stats['ppv'] = ppv
        if 'npv' in self.metrics:
            metrics = self.metrics['npv']
            npv = smp.metrics.negative_predictive_value(tp,fp,fn,tn,
                reduction=metrics['reduction'] if 'reduction' in metrics else 'micro',
                class_weights=metrics['class_weights'] if 'class_weights' in metrics else None,
                zero_division=metrics['zero_division'] if 'zero_division' in metrics else 1.0
            )
            self.metrics_stats['npv'] = npv
        if 'fnr' in self.metrics:
            metrics = self.metrics['fnr']
            fnr = smp.metrics.false_negative_rate(tp,fp,fn,tn,
                reduction=metrics['reduction'] if 'reduction' in metrics else 'micro',
                class_weights=metrics['class_weights'] if 'class_weights' in metrics else None,
                zero_division=metrics['zero_division'] if 'zero_division' in metrics else 1.0
            )
            self.metrics_stats['fnr'] = fnr
        if 'fpr' in self.metrics:
            metrics = self.metrics['fpr']
            fpr = smp.metrics.false_positive_rate(tp,fp,fn,tn,
                reduction=metrics['reduction'] if 'reduction' in metrics else 'micro',
                class_weights=metrics['class_weights'] if 'class_weights' in metrics else None,
                zero_division=metrics['zero_division'] if 'zero_division' in metrics else 1.0
            )
            self.metrics_stats['fpr'] = fpr
        if 'fdr' in self.metrics:
            metrics = self.metrics['fdr']
            fdr = smp.metrics.false_discovery_rate(tp,fp,fn,tn,
                reduction=metrics['reduction'] if 'reduction' in metrics else 'micro',
                class_weights=metrics['class_weights'] if 'class_weights' in metrics else None,
                zero_division=metrics['zero_division'] if 'zero_division' in metrics else 1.0
            )
            self.metrics_stats['fdr'] = fdr
        if 'for' in self.metrics:
            metrics = self.metrics['for']
            forv = smp.metrics.false_omission_rate(tp,fp,fn,tn,
                reduction=metrics['reduction'] if 'reduction' in metrics else 'micro',
                class_weights=metrics['class_weights'] if 'class_weights' in metrics else None,
                zero_division=metrics['zero_division'] if 'zero_division' in metrics else 1.0
            )
            self.metrics_stats['for'] = forv
        if 'lr+' in self.metrics:
            metrics = self.metrics['lr+']
            lrp = smp.metrics.positive_likelihood_ratio(tp,fp,fn,tn,
                reduction=metrics['reduction'] if 'reduction' in metrics else 'micro',
                class_weights=metrics['class_weights'] if 'class_weights' in metrics else None,
                zero_division=metrics['zero_division'] if 'zero_division' in metrics else 1.0
            )
            self.metrics_stats['lr+'] = lrp
        if 'lr-' in self.metrics:
            metrics = self.metrics['lr-']
            lrn = smp.metrics.negative_likelihood_ratio(tp,fp,fn,tn,
                reduction=metrics['reduction'] if 'reduction' in metrics else 'micro',
                class_weights=metrics['class_weights'] if 'class_weights' in metrics else None,
                zero_division=metrics['zero_division'] if 'zero_division' in metrics else 1.0
            )
            self.metrics_stats['lr-'] = lrn

    def compute(self,
        engine : Engine,
        experiment : Experiment,
        prefix : str = '',
        **kwargs
    ):
        self.log_metrics(engine, experiment, self.metrics_stats, prefix=prefix)