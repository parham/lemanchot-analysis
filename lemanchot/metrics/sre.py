
""" 
    @title A Deep Semi-supervised Segmentation Approach for Thermographic Analysis of Industrial Components
    @organization Laval University
    @professor  Professor Xavier Maldague
    @author     Parham Nooralishahi
    @email      parham.nooralishahi@gmail.com
"""

import numpy as np

from lemanchot.metrics.core import Function_Metric, assert_image_shapes_equal, metric_register

def sre(org: np.ndarray, pred: np.ndarray, **kwargs):
    """
    Signal to Reconstruction Error Ratio
    Based on: https://github.com/up42/image-similarity-measures
    """
    assert_image_shapes_equal(org, pred, "SRE")

    org_img = org if len(org.shape) > 2 else org.reshape((org.shape[0], org.shape[1], 1))
    pred_img = pred if len(pred.shape) > 2 else pred.reshape((pred.shape[0], pred.shape[1], 1))

    org_img = org_img.astype(np.float32)

    sre_final = []
    for i in range(org_img.shape[2]):
        numerator = np.square(np.mean(org_img[:, :, i]))
        denominator = (np.linalg.norm(org_img[:, :, i] - pred_img[:, :, i])) / (
            org_img.shape[0] * org_img.shape[1])
        sre_final.append(numerator / denominator)

    return {'sre' : 10 * np.log10(np.mean(sre_final))}

@metric_register('sre')
class SRE(Function_Metric):
    def __init__(self, config):
        super().__init__(
            func=sre,
            config=config
        )


