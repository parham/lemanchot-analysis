
""" 
    @project LeManchot-Analysis : Core components
    @organization Laval University
    @lab MiViM Lab
    @supervisor Professor Xavier Maldague
    @industrial-partner TORNGATS
"""

import numpy as np

from lemanchot.metrics.core import Function_Metric, assert_image_shapes_equal, metric_register 

def rmse(org: np.ndarray, pred: np.ndarray, max_p: int = 255, **kwargs) :
    """rmse : Root Mean Squared Error Calculated individually for all bands, then averaged
    Based on: https://github.com/up42/image-similarity-measures

    Args:
        org (np.ndarray): original image
        pred (np.ndarray): predicted image
        max_p (int, optional): maximum possible value. Defaults to 255.

    Returns:
        float: RMSE value
    """

    assert_image_shapes_equal(org, pred, "RMSE")

    rmse_bands = []
    tout = org if len(org.shape) > 2 else org.reshape((org.shape[0], org.shape[1], 1))
    tpred = pred if len(pred.shape) > 2 else pred.reshape((pred.shape[0], pred.shape[1], 1))
    for i in range(tout.shape[2]):
        dif = np.subtract(tout[:, :, i], tpred[:, :, i])
        m = np.mean(np.square(dif / max_p))
        s = np.sqrt(m)
        rmse_bands.append(s)

    return {'rmse' : np.mean(rmse_bands)}

@metric_register('rmse')
class RMSE(Function_Metric):
    def __init__(self, config):
        super().__init__(
            func=rmse,
            config=config
        )