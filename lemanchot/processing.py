
""" 
    @project LeManchot-Analysis : Core components
    @organization Laval University
    @lab MiViM Lab
    @supervisor Professor Xavier Maldague
    @industrial-partner TORNGATS
"""

import cv2
import torch
import numpy as np

from random import choice
from typing import Dict, List, Optional

def one_hot(labels: torch.Tensor,
            num_classes: int,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            eps: float = 1e-6) -> torch.Tensor:
    r"""
    Converts an integer label x-D tensor to a one-hot (x+1)-D tensor.
    Args:
        labels (torch.Tensor) : tensor with labels of shape :math:`(N, *)`,
                                where N is batch size. Each value is an integer
                                representing correct classification.
        num_classes (int): number of classes in labels.
        device (Optional[torch.device]): the desired device of returned tensor.
         Default: if None, uses the current device for the default tensor type
         (see torch.set_default_tensor_type()). device will be the CPU for CPU
         tensor types and the current CUDA device for CUDA tensor types.
        dtype (Optional[torch.dtype]): the desired data type of returned
         tensor. Default: if None, infers data type from values.
    Returns:
        torch.Tensor: the labels in one hot tensor of shape :math:`(N, C, *)`,
    Examples:
        >>> labels = torch.LongTensor([[[0, 1], [2, 0]]])
        >>> one_hot(labels, num_classes=3)
        tensor([[[[1.0000e+00, 1.0000e-06],
                  [1.0000e-06, 1.0000e+00]],
        <BLANKLINE>
                 [[1.0000e-06, 1.0000e+00],
                  [1.0000e-06, 1.0000e-06]],
        <BLANKLINE>
                 [[1.0000e-06, 1.0000e-06],
                  [1.0000e+00, 1.0000e-06]]]])
    """
    if not isinstance(labels, torch.Tensor):
        raise TypeError("Input labels type is not a torch.Tensor. Got {}"
                        .format(type(labels)))

    if not labels.dtype == torch.int64:
        raise ValueError(
            "labels must be of the same dtype torch.int64. Got: {}" .format(
                labels.dtype))

    if num_classes < 1:
        raise ValueError("The number of classes must be bigger than one."
                         " Got: {}".format(num_classes))

    shape = labels.shape
    one_hot = torch.zeros(
        (shape[0], num_classes) + shape[1:], device=device, dtype=dtype
    )

    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps

def extract_regions(
    data : np.ndarray, 
    min_size : int = 0
) -> List[Dict]:
    """ Extract independent regions from segmented image

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

def iou_binary(
    prediction : np.ndarray, 
    target : np.ndarray
):
    """ Measuring mean IoU metric for binary images

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

def adapt_output(
        output: np.ndarray,
        target: np.ndarray,
        iou_thresh: float = 0.1,
        use_null_class: bool = False):

    _, iou_map, maxv, selected_index, p_regs, t_regs = mIoU_func(
        output, target, iou_thresh=iou_thresh)

    labels = np.unique(target).tolist()
    
    null_class = choice([i for i in range(np.max(labels)+10) if i not in labels])
    maxv = np.amax(iou_map, axis=1).tolist()
    selected_index = np.argmax(iou_map, axis=1).tolist()
    result = np.zeros(output.shape, dtype=np.uint8)
    coupled = []
    for i in range(len(selected_index)):
        mv = maxv[i]
        # if mv > iou_thresh:
        preg = p_regs[i]
        treg = t_regs[selected_index[i]]
        classid = np.unique(treg).tolist()
        if len(classid) > 1:
            classid = classid[-1]
            if use_null_class:
                result[preg > 0] = classid if mv > iou_thresh else null_class
                coupled.append((preg, treg))
            elif mv > iou_thresh:
                result[preg > 0] = classid
                coupled.append((preg, treg))

    return result, iou_map, coupled

def regions_to_image(regions : List[np.ndarray]) -> np.ndarray:
    if len(regions) == 0:
        return
    res = None
    for r in regions:
        res = res + r if res is not None else r
    return res

def remove_small_regions(img : np.ndarray, min_area : int = 0):
    regs = extract_regions(img)
    regs = list(filter(lambda x : np.count_nonzero(x) > min_area, regs))
    return regions_to_image(regs)
