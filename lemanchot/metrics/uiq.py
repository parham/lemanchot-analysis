
""" 
    @project LeManchot : Multi-Modal Data Acquisition and Processing of Drone-based Inspection
    @organization Laval University
    @lab MiViM Lab
    @supervisor Professor Xavier Maldague
    @industrial-partner TORNGATS
"""

import numpy as np

from lemanchot.metrics.core import Function_Metric, assert_image_shapes_equal, metric_register

def sliding_window(image: np.ndarray, stepSize: int, windowSize: int):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y : y + windowSize[1], x : x + windowSize[0]])

def uiq (org: np.ndarray, 
    pred: np.ndarray, 
    step_size: int = 1, 
    window_size: int = 8,
    **kwargs
):
    """
    Universal Image Quality index
    Based on: https://github.com/up42/image-similarity-measures
    """
    # TODO: Apply optimization, right now it is very slow
    assert_image_shapes_equal(org, pred, "UIQ")

    org = org.astype(np.float32)
    pred = pred.astype(np.float32)

    org_img = org if len(org.shape) > 2 else org.reshape((org.shape[0], org.shape[1], 1))
    pred_img = pred if len(pred.shape) > 2 else pred.reshape((pred.shape[0], pred.shape[1], 1))

    q_all = []
    for (x, y, window_org), (x, y, window_pred) in zip(
        sliding_window(org_img, stepSize=step_size, windowSize=(window_size, window_size)),
        sliding_window(pred_img, stepSize=step_size, windowSize=(window_size, window_size)),
    ):
        # if the window does not meet our desired window size, ignore it
        if window_org.shape[0] != window_size or window_org.shape[1] != window_size:
            continue

        for i in range(org_img.shape[2]):
            org_band = window_org[:, :, i]
            pred_band = window_pred[:, :, i]
            org_band_mean = np.mean(org_band)
            pred_band_mean = np.mean(pred_band)
            org_band_variance = np.var(org_band)
            pred_band_variance = np.var(pred_band)
            org_pred_band_variance = np.mean(
                (org_band - org_band_mean) * (pred_band - pred_band_mean)
            )

            numerator = 4 * org_pred_band_variance * org_band_mean * pred_band_mean
            denominator = (org_band_variance + pred_band_variance) * (
                org_band_mean ** 2 + pred_band_mean ** 2
            )

            if denominator != 0.0:
                q = numerator / denominator
                q_all.append(q)

    if not np.any(q_all):
        raise ValueError(
            f"Window size ({window_size}) is too big for image with shape "
            f"{org_img.shape[0:2]}, please use a smaller window size."
        )

    return {'uiq' : np.mean(q_all)}

@metric_register('uiq')
class UIQ(Function_Metric):
    def __init__(self, config):
        super().__init__(
            func=uiq,
            config=config
        )