
""" 
    @project LeManchot : Multi-Modal Data Acquisition and Processing of Drone-based Inspection
    @organization Laval University
    @lab MiViM Lab
    @supervisor Professor Xavier Maldague
    @industrial-partner TORNGATS
"""

import numpy as np

from lemanchot.metrics.core import Function_Metric, assert_image_shapes_equal, metric_register 

def psnr(org_img: np.ndarray, pred_img: np.ndarray, max_p: int = 255, **kwargs) -> float:
    """
    Peek Signal to Noise Ratio, implemented as mean squared error converted to dB.
    Based on: https://github.com/up42/image-similarity-measures

    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)
    When using 12-bit imagery MaxP is 4095, for 8-bit imagery 255. For floating point imagery using values between
    0 and 1 (e.g. unscaled reflectance) the first logarithmic term can be dropped as it becomes 0
    """
    assert_image_shapes_equal(org_img, pred_img, "PSNR")

    mse_bands = []
    org = org_img if len(org_img.shape) > 2 else org_img.reshape((org_img.shape[0], org_img.shape[1], 1))
    pred = pred_img if len(pred_img.shape) > 2 else pred_img.reshape((pred_img.shape[0], pred_img.shape[1], 1))
    for i in range(org.shape[2]):
        mse_bands.append(np.mean(np.square(org[:, :, i] - pred[:, :, i])))

    return {'psnr' : 20 * np.log10(max_p) - 10.0 * np.log10(np.mean(mse_bands))}

@metric_register('psnr')
class PSNR(Function_Metric):
    def __init__(self, config):
        super().__init__(
            func=psnr,
            config=config
        )