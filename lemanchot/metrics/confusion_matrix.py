
""" 
    @project LeManchot-Analysis : Core components
    @organization Laval University
    @lab MiViM Lab
    @supervisor Professor Xavier Maldague
    @industrial-partner TORNGATS
"""

import numpy as np

from dataclasses import dataclass
from typing import Dict, List

from comet_ml import Experiment

from ignite.engine import Engine

from lemanchot.metrics import BaseMetric, metric_register

@dataclass
class CMRecord:
    """ Confusion Matrix """
    confusion_matrix : np.ndarray
    step_confusion_matrix : np.ndarray
    class_labels : List[str]
    cm_metrics : Dict[str, float]

def measure_accuracy_cm__(
    cmatrix : np.ndarray
) -> Dict:
    """Measuring accuracy metrics

    Args:
        cmatrix (np.ndarray): confusion matrix
        labels (List[str]): _description_

    Returns:
        Dict: _description_
    """
    fp = cmatrix.sum(axis=0) - np.diag(cmatrix)  
    fn = cmatrix.sum(axis=1) - np.diag(cmatrix)
    tp = np.diag(cmatrix)
    tn = cmatrix.sum() - (fp + fn + tp)

    # Calculate statistics
    accuracy = np.nan_to_num(np.diag(cmatrix) / cmatrix.sum())
    precision = np.nan_to_num(tp / (tp + fp))
    recall = np.nan_to_num(tp / (tp + fn))

    tmp1 = 2 * precision * recall
    tmp2 = precision + recall
    fscore = np.divide(tmp1, tmp2, out=np.zeros_like(tmp1), where=tmp2 != 0)
    # Calculate weights
    weights = cmatrix.sum(axis=0)
    weights = weights / weights.sum()

    return {
        'precision' : np.average(precision, weights=weights),
        'recall' : np.average(recall, weights=weights),
        'accuracy' : np.average(accuracy, weights=weights),
        'fscore' : np.average(fscore, weights=weights),
    }

@metric_register('confusion_matrix')
class ConfusionMatrix(BaseMetric):
    def __init__(self, config) -> None:
        super().__init__(config)
        lbl = list(self.categories.values())
        lbl.sort()
        self.class_ids = lbl
        self.class_labels = [list(self.categories.keys())[self.class_ids.index(v)] for v in self.class_ids]
        self.reset()

    def reset(self):
        """ Reset the internal metrics """
        lcount = len(self.categories.keys())
        self.confusion_matrix = np.zeros((lcount, lcount), np.uint)
        self.step_confusion_matrix = np.zeros((lcount, lcount), np.uint)

    def expand_by_one(self):
        row, col = self.confusion_matrix.shape
        # Add a column
        c = np.zeros((row,1))
        newc = np.hstack((self.confusion_matrix, c))
        newc_step = np.hstack((self.confusion_matrix, c))
        # Add a row
        r = np.zeros((1, col + 1))
        newc = np.vstack((newc,r))
        newc_step = np.vstack((newc_step,r))
        self.confusion_matrix = newc
        self.step_confusion_matrix = newc_step

    def update(self, data, **kwargs):
        output, target = data[-2], data[-1]
        # Flattening the output and target
        out = output.flatten()
        tar = target.flatten()
        tar_inds = np.unique(tar)
        out = out.tolist()
        tar = tar.tolist()
        # Check if there are missing values in target
        for ind in tar_inds:
            if not ind in self.class_ids:
                self.class_ids.append(ind)
                self.class_labels.append(f'Unknow_{ind}')
                self.expand_by_one()
        # Update Confusion Matrix
        cmatrix = np.zeros(self.confusion_matrix.shape, np.uint)
        for i in range(len(out)):
            o, t = out[i], tar[i]
            if o in self.class_ids:
                oind = self.class_ids.index(o)
                tind = self.class_ids.index(t)
                cmatrix[tind, oind] += 1
        self.step_confusion_matrix = cmatrix
        self.confusion_matrix += cmatrix
    
    def compute(self,  
        engine : Engine,
        experiment : Experiment,
        prefix : str = '',
        **kwargs
    ):
        experiment.log_confusion_matrix(
            matrix=self.confusion_matrix, 
            labels=self.class_labels, 
            title=f'{prefix}Confusion Matrix',
            file_name=f'{prefix}confusion-matrix.json', 
            step=engine.state.iteration, 
            epoch=engine.state.epoch
        )
        
        # Calculate confusion matrix based metrics
        stats = measure_accuracy_cm__(self.confusion_matrix)
        self.log_metrics(engine, experiment, stats, prefix=prefix)

        return CMRecord(
            self.confusion_matrix,
            self.step_confusion_matrix,
            self.class_labels,
            cm_metrics=stats
        )