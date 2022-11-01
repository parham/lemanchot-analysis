"""
    @project LeManchot-Analysis : Core components
    @organization Laval University
    @lab MiViM Lab
    @supervisor Professor Xavier Maldague
    @industrial-partner TORNGATS
"""
import os
import json
import time
from itertools import product
from typing import Callable, Dict, List, NoReturn, Optional, Tuple, T as Array

import numpy as np
import torch
from ignite.engine import Engine
from lemanchot.core import make_tensor_for_comet
from lemanchot.models import BaseModule
from lemanchot.pipeline.core import get_device, get_profile, pipeline_register
from lemanchot.pipeline.wrapper import BaseWrapper, wrapper_register

from sheldon_data_stages.stages.rle.handler_rle import generateJSON


def sliding_slices(size: int, kernel: int, stride: Optional[int] = None) -> slice:
    """sliding_slices.
    Args:
        size (int): Max size of the conteiner.
        kernel (int): Sliding window size.
        overlap (Optional[int]): Stride of the kernel.
    Returns:
        slice: Slided slice.
    Note: If the size is not perfectly divisible by the kernel size, the
          last slice will be corrected to finish perfectly on the size value.
    """
    k1 = kernel // 2
    k2 = kernel - k1

    pt = k1
    while pt + k2 < size:
        yield slice(pt - k1, pt + k2)
        pt += stride

    # Add last point to sliding
    pt = size - k2
    yield slice(pt - k1, pt + k2)


def sliding_window(
    array_shape: List[int],
    axis: List[int],
    kernel: Tuple[int],
    stride: Optional[Tuple[int]] = (None, None),
) -> Array:
    """sliding_window.
    Args:
        X (Array): Input array of any size.
        axis (Tuple[int]): Axis of X to be sliced.
        kernel (Tuple[int]): Size of the sliding window.
        overlap (Optional[Tuple[int]]): Stride of the window.
    Returns:
        Array: Sliced patch of the input X.
    """
    sizes = (array_shape[ax] for ax in axis)
    windows = [list(sliding_slices(s, k, o)) for s, k, o in zip(sizes, kernel, stride)]

    # Generate default slices list, here slice(None) is syntax sugar for `:` which selects
    # the whole dimension.
    default_slice = np.array([slice(None) for _ in array_shape])
    # FIXME: product consuming all the windows generators,
    # hence preconsuming them is faster.
    for slc in product(*windows):
        default_slice[axis] = slc
        tuple_slices = tuple(default_slice)
        yield tuple_slices


@wrapper_register("default_predict")
class PredictWrapper(BaseWrapper):
    def __init__(self, step_func: Callable, device: str, **kwargs) -> None:
        super().__init__(step_func, device, **kwargs)

    def __call__(self, engine: Engine, batch) -> NoReturn:

        # Local Variable Initialization
        step_func = self.step_func
        device = self.device
        model = self.model
        img_saver = self.img_saver if hasattr(self, "img_saver") else None

        profile = get_profile(engine.state.profile_name)
        paths, data = batch
        # Logging computation time
        t = time.time()
        # Apply the model to data
        res = step_func(
            engine=engine,
            batch=data.to(device=get_device()),
            device=device,
            model=model,
        )
        step_time = time.time() - t

        # Logging loss & step time
        if "loss" in res:
            engine.state.metrics["loss"] = res["loss"]
        engine.state.metrics["step_time"] = step_time

        if profile.enable_logging and img_saver is not None:
            # Assume Tensor B x C x W x H
            # Logging imagery results
            for name, img in zip(paths, res["y_pred"]):
                sample = make_tensor_for_comet(img)
                label = os.path.basename(name).split('.')[0]
                img_saver(label, sample)
                ohe = torch.where(img[0, 1:, ...] != 0, 1, 0).type(torch.uint8).cpu().numpy()
                data = generateJSON(
                        ohe,
                        {
                            "fileID": name,
                            "classes": ["crack"],
                        },
                    )
                with open(f"/data/MetroPanama/results/{label}.json", "w") as f:
                    json.dump(data, f)



@pipeline_register("simple_predict")
@torch.no_grad()
def simple_predict_step__(engine: Engine, batch, model: BaseModule, **kwargs) -> Dict:

    inputs = batch

    model.eval()
    outputs = model(inputs)
    outputs = outputs.argmax(dim=1, keepdims=True)

    return {"y_pred": outputs}


@pipeline_register("simple_multilabel_predict")
@torch.no_grad()
def simple_multilabel_step__(
    engine: Engine, batch, model: BaseModule, **kwargs
) -> Dict:

    n, c, h, w = batch.shape
    outputs = torch.zeros((n, 2, h, w), device=batch.device)
    model.eval()
    for s in sliding_window(batch.shape, axis=[-2, -1], kernel=(512, 512), stride=(512, 512)):
        outputs[s] = model(batch[s])

    outputs = torch.threshold(outputs.sigmoid(), 0.5, 0)

    return {"y_pred": outputs}
