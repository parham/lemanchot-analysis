
""" 
    @project LeManchot : Multi-Modal Data Acquisition and Processing of Drone-based Inspection
    @organization Laval University
    @lab MiViM Lab
    @supervisor Professor Xavier Maldague
    @industrial-partner TORNGATS
"""

import numpy as np

from lemanchot.metrics.core import Function_Metric, assert_image_shapes_equal, metric_register

def sam(org: np.ndarray, pred: np.ndarray, convert_to_degree: bool = True, **kwargs):
    """
    Spectral Angle Mapper which defines the spectral similarity between two spectra
    Based on: https://github.com/up42/image-similarity-measures
    """
    assert_image_shapes_equal(org, pred, "SAM")

    org_img = org if len(org.shape) > 2 else org.reshape((org.shape[0], org.shape[1], 1))
    pred_img = pred if len(pred.shape) > 2 else pred.reshape((pred.shape[0], pred.shape[1], 1))

    # Spectral angles are first computed for each pair of pixels
    numerator = np.sum(np.multiply(pred_img, org_img), axis=2)
    denominator = np.linalg.norm(org_img, axis=2) * np.linalg.norm(pred_img, axis=2)
    val = np.clip(numerator / denominator, -1, 1)
    sam_angles = np.arccos(val)
    if convert_to_degree:
        sam_angles = sam_angles * 180.0 / np.pi

    # The original paper states that SAM values are expressed as radians, while e.g. Lanares
    # et al. (2018) use degrees. We therefore made this configurable, with degree the default
    return {'sam' : np.mean(np.nan_to_num(sam_angles))}

@metric_register('sam')
class SAM(Function_Metric):
    def __init__(self, config):
        super().__init__(
            func=sam,
            config=config
        )

