
""" 
    @project LeManchot : Multi-Modal Data Acquisition and Processing of Drone-based Inspection
    @organization Laval University
    @lab MiViM Lab
    @supervisor Professor Xavier Maldague
    @industrial-partner TORNGATS
"""

import numpy as np

from skimage.metrics import structural_similarity

from lemanchot.metrics.core import Function_Metric, assert_image_shapes_equal, metric_register 

@metric_register('ssim')
class SSIM(Function_Metric):
    def __init__(self, config):
        super().__init__(
            func=ssim,
            config=config
        )

def ssim(org_img: np.ndarray, pred_img: np.ndarray, max_p: int = 255, **kwargs) -> float:
    """
    Structural Simularity Index
    Based on: https://github.com/up42/image-similarity-measures
    """
    assert_image_shapes_equal(org_img, pred_img, "SSIM")

    res = structural_similarity(org_img, pred_img, data_range=max_p, multichannel=True)
    return {'ssim' : res}

def _ehs(x: np.ndarray, y: np.ndarray):
    """
    Entropy-Histogram Similarity measure
    """
    H = (np.histogram2d(x.flatten(), y.flatten()))[0]

    return -np.sum(np.nan_to_num(H * np.log2(H)))

def _edge_c(x: np.ndarray, y: np.ndarray):
    """
    Edge correlation coefficient based on Canny detector
    """
    # Use 100 and 200 as thresholds, no indication in the paper what was used
    g = cv2.Canny((x * 0.0625).astype(np.uint8), 100, 200)
    h = cv2.Canny((y * 0.0625).astype(np.uint8), 100, 200)

    g0 = np.mean(g)
    h0 = np.mean(h)

    numerator = np.sum((g - g0) * (h - h0))
    denominator = np.sqrt(np.sum(np.square(g - g0)) * np.sum(np.square(h - h0)))

    return numerator / denominator

def issm(org: np.ndarray, pred: np.ndarray, **kwargs) -> float:
    """
    Information theoretic-based Statistic Similarity Measure
    Note that the term e which is added to both the numerator as well as the denominator is not properly
    introduced in the paper. We assume the authers refer to the Euler number.
    Based on: https://github.com/up42/image-similarity-measures
    """
    assert_image_shapes_equal(org, pred, "ISSM")

    org_img = org if len(org.shape) > 2 else org.reshape((org.shape[0], org.shape[1], 1))
    pred_img = pred if len(pred.shape) > 2 else pred.reshape((pred.shape[0], pred.shape[1], 1))

    # Variable names closely follow original paper for better readability
    x = org_img
    y = pred_img
    A = 0.3
    B = 0.5
    C = 0.7

    ehs_val = _ehs(x, y)
    canny_val = _edge_c(x, y)

    numerator = canny_val * ehs_val * (A + B) + math.e
    denominator = A * canny_val * ehs_val + B * ehs_val + C * ssim(x, y) + math.e

    return {'issm' : np.nan_to_num(numerator / denominator)}

@metric_register('issm')
class ISSM(Function_Metric):
    def __init__(self, config):
        super().__init__(
            func=issm,
            config=config
        )