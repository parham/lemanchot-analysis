
import logging
import cv2
import numpy as np

from dataclasses import dataclass
from typing import Callable, Dict, List
from skimage.metrics import structural_similarity
from scipy.spatial.distance import directed_hausdorff
from comet_ml import Experiment
from dotmap import DotMap

import phasepack.phasecong as pc

from ignite.engine import Engine
from ignite.exceptions import NotComputableError

__metric_handler = {}

def metric_register(name : str):
    """ register metrics to be used """
    def __embed_func(clss):
        global __metric_handler
        if not issubclass(clss, BaseMetric):
            raise NotImplementedError('The specified metric is not correctly implemented!')

        clss.get_name = lambda _: name
        __metric_handler[name] = clss

    return __embed_func

def list_metrics() -> List[str]:
    """List of registered models

    Returns:
        List[str]: list of registered models
    """
    global __metric_handler
    return list(__metric_handler.keys())

class BaseMetric(object):
    """ Base class for metrics """
    def __init__(self, config):
        # Initialize the configuration
        for key, value in config.items():
            setattr(self, key, value)

    def reset(self):
        """ reset the internal states """
        return

    def update(self, batch, **kwargs):
        """ update the internal states with given output

        Args:
            batch (Tuple): the variable containing data
        """
        pass

    def compute(self,
        engine : Engine,
        experiment : Experiment,
        prefix : str = '',
        **kwargs
    ):
        """ Compute the metrics

        Args:
            prefix (str, optional): prefix for logging metrics. Defaults to ''.
            step (int, optional): the given step. Defaults to 1.
            epoch (int, optional): the given epoch. Defaults to 1.
        """
        pass

def load_metric(name, config) -> BaseMetric:
    if not name in list_metrics():
        msg = f'{name} metric is not supported!'
        logging.error(msg)
        raise ValueError(msg)
    
    return __metric_handler[name](config)

def load_metrics(experiment_config : DotMap, categories):
    metrics_configs = experiment_config.metrics

    metrics_obj = []
    for metric_name, config in metrics_configs.items():
        config.categories = categories
        metrics_obj.append(load_metric(metric_name, config))
    return metrics_obj

@dataclass
class CMRecord:
    """ Confusion Matrix """
    confusion_matrix : np.ndarray
    step_confusion_matrix : np.ndarray
    class_labels : List[str]
    cm_metrics : Dict[str, float]

class Function_Metric(BaseMetric):
    """
        Function_Metric is a metric class that allows you to wrap a metric function inside.
        It lets a metric function to be integrated into the prepared platform.
    """
    def __init__(self, 
        func : Callable, 
        config
    ):
        super().__init__(config)
        self.__func = func
        self.__last_ret = None
        self.__args = config

    def update(self, batch, **kwargs):
        """ update the internal states with given batch """
        output, target = batch[-2], batch[-1]
        self.__last_ret = self.__func(output, target, **self.__args)

    def compute(self,
        engine : Engine,
        experiment : Experiment,
        prefix : str = '',
        **kwargs
    ):
        """Compute the metrics

        Args:
            prefix (str, optional): prefix for logging metrics. Defaults to ''.
            step (int, optional): the given step. Defaults to 1.
            epoch (int, optional): the given epoch. Defaults to 1.
        """
        
        if self.__last_ret is not None:
            experiment.log_metrics(
                self.__last_ret, 
                prefix=prefix, 
                step=engine.state.iteration, 
                epoch=engine.state.epoch
            )
        
        return self.__last_ret

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
        stats = {}

        sts = measure_accuracy_cm__(self.confusion_matrix)
        stats = {**stats, **sts}
        
        experiment.log_metrics(stats, 
            prefix=prefix, 
            step=engine.state.iteration, 
            epoch=engine.state.epoch
        )

        return CMRecord(
            self.confusion_matrix,
            self.step_confusion_matrix,
            self.class_labels,
            cm_metrics=stats
        )

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

def iou_binary(
    prediction : np.ndarray, 
    target : np.ndarray
):
    """Measuring mean IoU metric for binary images

    Args:
        prediction (np.ndarray): The image containing the prediction
        target (np.ndarray): The image containing the target

    Returns:
        float: the mean of IoU across the IoU of all regions
    """

    # Calculate intersection
    intersection = np.count_nonzero(np.logical_and(prediction, target))
    # Calculate union
    union = np.count_nonzero(np.logical_or(prediction, target))
    # Calculate IoU
    iou = float(intersection) / float(union) if union != 0 else 0
    return iou

def extract_regions(
    data : np.ndarray, 
    min_size : int = 0
) -> List[Dict]:
    """Extract independent regions from segmented image

    Args:
        data (np.ndarray): segmented image which each pixel presented the class id.

    Returns:
        List[Dict]: List of dictionary where each item has two key item: 
            (a) 'class' : the class id associated to the region, 
            (b) 'region' : the extracted isolated region. The region blob is binalized so the value is {0,1}.
    """
    
    # Determine the number of class labels
    labels = np.unique(data).tolist()
    if len(labels) < 2:
        return [data]

    result = []
    for i in range(1, len(labels)):
        clss_id = labels[i]
        class_layer = data * (data == clss_id)
        numLabels, area, _, _ = cv2.connectedComponentsWithStats(class_layer, 8)
        for j in range(1, numLabels):
            region = data * (area == j)
            if np.count_nonzero(region) > min_size:
                result.append(region)

    return result

def mIoU_func(
    output : np.ndarray, 
    target : np.ndarray, 
    iou_thresh : float = 0,
    **kwargs):
    """ Measuring mean IoU

    Args:
        prediction (np.ndarray): The image containing the prediction
        target (np.ndarray): The image containing the target
        iou_thresh (float, optional): The threshold to filter out IoU measures. Defaults to 0.
        details (bool, optional): Determines whether the function's return contains the detailed results or not! Defaults to False.

    Returns:
        float: mean IoU
        numpy.ndarray : the table containing the IoU values for each region in target and prediction.
    """
    p_regs = extract_regions(output, min_size=10)
    t_regs = extract_regions(target)
    # b. Calculate the IoU map of prediction-region map
    # b.1. Create a matrix n_p x n_t (M) ... rows are predictions and columns are targets
    p_count = len(p_regs)
    t_count = len(t_regs)
    iou_map = np.zeros((p_count, t_count))
    for pid in range(p_count):
        p_bin = p_regs[pid] > 0
        for tid in range(t_count):
            t_bin = t_regs[tid] > 0
            iou_map[pid,tid] = iou_binary(p_bin, t_bin) 
    
    max_iou = np.amax(iou_map, axis=1)
    max_iou_index = np.argmax(iou_map, axis=1)
    iou = np.mean(max_iou[max_iou > iou_thresh])

    return iou, iou_map, max_iou.tolist(), max_iou_index.tolist(), p_regs, t_regs

def _assert_image_shapes_equal(org: np.ndarray, pred: np.ndarray, metric: str):
    """ Shape of the image should be like this (rows, cols, bands)
        Please note that: The interpretation of a 3-dimension array read from rasterio is: (bands, rows, columns) while
        image processing software like scikit-image, pillow and matplotlib are generally ordered: (rows, columns, bands)
        in order efficiently swap the axis order one can use reshape_as_raster, reshape_as_image from rasterio.plot
    
    Based on: https://github.com/up42/image-similarity-measures
    Args:
        org_img (np.ndarray): original image
        pred_img (np.ndarray): predicted image
        metric (str): _description_
    """

    msg = (
        f"Cannot calculate {metric}. Input shapes not identical. y_true shape ="
        f"{str(org.shape)}, y_pred shape = {str(pred.shape)}"
    )

    assert org.shape == pred.shape, msg

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

    _assert_image_shapes_equal(org, pred, "RMSE")

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

def psnr(org_img: np.ndarray, pred_img: np.ndarray, max_p: int = 255, **kwargs) -> float:
    """
    Peek Signal to Noise Ratio, implemented as mean squared error converted to dB.
    Based on: https://github.com/up42/image-similarity-measures

    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)
    When using 12-bit imagery MaxP is 4095, for 8-bit imagery 255. For floating point imagery using values between
    0 and 1 (e.g. unscaled reflectance) the first logarithmic term can be dropped as it becomes 0
    """
    _assert_image_shapes_equal(org_img, pred_img, "PSNR")

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

def _similarity_measure(x: np.array, y: np.array, constant: float):
    """
    Calculate feature similarity measurement between two images
    """
    numerator = 2 * x * y + constant
    denominator = x ** 2 + y ** 2 + constant

    return numerator / denominator

def _gradient_magnitude(img: np.ndarray, img_depth: int):
    """
    Calculate gradient magnitude based on Scharr operator.
    """
    res = 0
    try:
        scharrx = cv2.Scharr(img, img_depth, 1, 0)
        scharry = cv2.Scharr(img, img_depth, 0, 1)
        res = np.sqrt(scharrx ** 2 + scharry ** 2)
    except Exception:
        res = 0
    return res

def directed_hausdorff_distance(img: np.ndarray, target: np.ndarray, **kwargs):
    hdvalue = max(directed_hausdorff(img, target)[0], directed_hausdorff(target, img)[0])
    return {'directed_hausdorff' : hdvalue}

@metric_register('hausdorff_distance')
class Hausdorff_Distance(Function_Metric):
    def __init__(self, config):
        super().__init__(
            func=directed_hausdorff_distance,
            config=config
        )

def fsim(
    org: np.ndarray, 
    pred: np.ndarray, 
    T1: float = 0.85, T2: float = 160,
    **kwargs
) -> float:
    """
    Based on: https://github.com/up42/image-similarity-measures
    Feature-based similarity index, based on phase congruency (PC) and image gradient magnitude (GM)
    There are different ways to implement PC, the authors of the original FSIM paper use the method
    defined by Kovesi (1999). The Python phasepack project fortunately provides an implementation
    of the approach.
    There are also alternatives to implement GM, the FSIM authors suggest to use the Scharr
    operation which is implemented in OpenCV.
    Note that FSIM is defined in the original papers for grayscale as well as for RGB images. Our use cases
    are mostly multi-band images e.g. RGB + NIR. To accommodate for this fact, we compute FSIM for each individual
    band and then take the average.
    Note also that T1 and T2 are constants depending on the dynamic range of PC/GM values. In theory this parameters
    would benefit from fine-tuning based on the used data, we use the values found in the original paper as defaults.
    Args:
        org_img -- numpy array containing the original image
        pred_img -- predicted image
        T1 -- constant based on the dynamic range of PC values
        T2 -- constant based on the dynamic range of GM values
    """
    _assert_image_shapes_equal(org, pred, "FSIM")

    org_img = org if len(org.shape) > 2 else org.reshape((org.shape[0], org.shape[1], 1))
    pred_img = pred if len(pred.shape) > 2 else pred.reshape((pred.shape[0], pred.shape[1], 1))

    alpha = (beta) = 1  # parameters used to adjust the relative importance of PC and GM features
    fsim_list = []
    for i in range(org_img.shape[2]):
        # Calculate the PC for original and predicted images
        pc1_2dim = pc(org_img[:, :, i], nscale=4, minWaveLength=6, mult=2, sigmaOnf=0.5978)
        pc2_2dim = pc(pred_img[:, :, i], nscale=4, minWaveLength=6, mult=2, sigmaOnf=0.5978)

        # pc1_2dim and pc2_2dim are tuples with the length 7, we only need the 4th element which is the PC.
        # The PC itself is a list with the size of 6 (number of orientation). Therefore, we need to
        # calculate the sum of all these 6 arrays.
        pc1_2dim_sum = np.zeros((org_img.shape[0], org_img.shape[1]), dtype=np.float64)
        pc2_2dim_sum = np.zeros((pred_img.shape[0], pred_img.shape[1]), dtype=np.float64)
        for orientation in range(6):
            pc1_2dim_sum += pc1_2dim[4][orientation]
            pc2_2dim_sum += pc2_2dim[4][orientation]

        # Calculate GM for original and predicted images based on Scharr operator
        gm1 = _gradient_magnitude(org_img[:, :, i], cv2.CV_16U)
        gm2 = _gradient_magnitude(pred_img[:, :, i], cv2.CV_16U)

        # Calculate similarity measure for PC1 and PC2
        S_pc = _similarity_measure(pc1_2dim_sum, pc2_2dim_sum, T1)
        # Calculate similarity measure for GM1 and GM2
        S_g = _similarity_measure(gm1, gm2, T2)

        S_l = (S_pc ** alpha) * (S_g ** beta)

        numerator = np.sum(S_l * np.maximum(pc1_2dim_sum, pc2_2dim_sum))
        denominator = np.sum(np.maximum(pc1_2dim_sum, pc2_2dim_sum))
        fsim_list.append(numerator / denominator)

    return {'fsim' : np.mean(fsim_list)}

@metric_register('fsim')
class FSIM(Function_Metric):
    def __init__(self, config):
        super().__init__(
            func=fsim,
            config=config
        )

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
    _assert_image_shapes_equal(org_img, pred_img, "SSIM")

    res = structural_similarity(org_img, pred_img, data_range=max_p, multichannel=True)
    return {'ssim' : res}

@metric_register('issm')
class ISSM(Function_Metric):
    def __init__(self, config):
        super().__init__(
            func=issm,
            config=config
        )

def issm(org: np.ndarray, pred: np.ndarray, **kwargs) -> float:
    """
    Information theoretic-based Statistic Similarity Measure
    Note that the term e which is added to both the numerator as well as the denominator is not properly
    introduced in the paper. We assume the authers refer to the Euler number.
    Based on: https://github.com/up42/image-similarity-measures
    """
    _assert_image_shapes_equal(org, pred, "ISSM")

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

def sliding_window(image: np.ndarray, stepSize: int, windowSize: int):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y : y + windowSize[1], x : x + windowSize[0]])

@metric_register('uiq')
class UIQ(Function_Metric):
    def __init__(self, config):
        super().__init__(
            func=uiq,
            config=config
        )

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
    _assert_image_shapes_equal(org, pred, "UIQ")

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

@metric_register('sam')
class SAM(Function_Metric):
    def __init__(self, config):
        super().__init__(
            func=sam,
            config=config
        )

def sam(org: np.ndarray, pred: np.ndarray, convert_to_degree: bool = True, **kwargs):
    """
    Spectral Angle Mapper which defines the spectral similarity between two spectra
    Based on: https://github.com/up42/image-similarity-measures
    """
    _assert_image_shapes_equal(org, pred, "SAM")

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

@metric_register('sre')
class SRE(Function_Metric):
    def __init__(self, config):
        super().__init__(
            func=sre,
            config=config
        )

def sre(org: np.ndarray, pred: np.ndarray, **kwargs):
    """
    Signal to Reconstruction Error Ratio
    Based on: https://github.com/up42/image-similarity-measures
    """
    _assert_image_shapes_equal(org, pred, "SRE")

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
