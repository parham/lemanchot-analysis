"""
    @project LeManchot-Analysis : Core components
    @organization Laval University
    @lab MiViM Lab
    @supervisor Professor Xavier Maldague
    @industrial-partner TORNGATS
"""

import os
import time
import logging
import functools

from dotmap import DotMap
from comet_ml import Experiment
from typing import Callable, Dict, List, Union

import numpy as np
import torch
import torch.optim as optim
from torchvision.transforms import ToPILImage

from ignite.engine import Engine
from ignite.engine.events import Events
from ignite.handlers import ModelCheckpoint, global_step_from_engine

from lemanchot.core import (
    exception_logger,
    get_config,
    get_device,
    get_experiment,
    get_profile,
    load_settings,
    make_tensor_for_comet,
)
from lemanchot.loss import load_loss
from lemanchot.metrics import BaseMetric, load_metrics
from lemanchot.models import BaseModule, load_model

def load_optimizer(model: BaseModule, experiment_config: DotMap) -> optim.Optimizer:
    """Load the optimizer based on given configuration

    Args:
        model (BaseModule): the model
        experiment_config (DotMap): configuration for optimizer

    Returns:
        optim.Optimizer: the instantiated optimizer
    """

    if model is None or \
        not "optimizer" in experiment_config:
        return None

    params = model.parameters()

    optim_name = experiment_config.optimizer.name
    optim_config = (
        experiment_config.optimizer.config
        if "config" in experiment_config.optimizer else {}
    )

    return {
        "SGD": lambda ps, config: optim.SGD(ps, **config),
        "Adam": lambda ps, config: optim.Adam(ps, **config),
    }[optim_name](params, optim_config)

# def load_scheduler(
#     optimizer: optim.Optimizer, experiment_config: DotMap
# ) -> optim._LRScheduler:
#     """Load the optimizer based on given configuration

#     Args:
#         model (BaseModule): the model
#         experiment_config (DotMap): configuration for optimizer

#     Returns:
#         optim.Optimizer: the instantiated optimizer
#     """

#     if optimizer is None:
#         return None

#     if not "scheduler" in experiment_config:
#         return None

#     sch_name = experiment_config.scheduler.name
#     sch_config = (
#         experiment_config.scheduler.config
#         if "config" in experiment_config.scheduler
#         else {}
#     )

#     return {
#         "ReduceLROnPlateau": lambda ps, config: optim.SGD(ps, **config),
#         "CosineAnnealingLR": lambda ps, config: optim.Adam(ps, **config),
#     }[sch_name](optimizer, sch_config)

__pipeline_handler = {}

def pipeline_register(name: Union[str, List[str]]):
    """Register a pipeline into the pipeline repository

    Args:
        name (Union[str, List[str]]): the pipeline's or list of pipeline's name(s)
    """

    def __embed_func(func):
        global __pipeline_handler
        hname = name if isinstance(name, list) else [name]
        for n in hname:
            __pipeline_handler[n] = func

    return __embed_func

def list_pipelines() -> List[str]:
    """List of registered pipeline

    Returns:
        List[str]: list of registered pipelines
    """
    global __pipeline_handler
    return list(__pipeline_handler.keys())

@exception_logger
def load_pipeline(pipeline_name: str) -> Callable:
    """Load pipeline based on the given name

    Args:
        pipeline_name (str): pipeline's name

    Raises:
        ValueError: if pipeline's name is not supported

    Returns:
        Callable: the pipeline handler function
    """

    global __pipeline_handler
    if not pipeline_name in list_pipelines():
        msg = f"{pipeline_name} model is not supported!"
        logging.error(msg)
        raise ValueError(msg)

    return __pipeline_handler[pipeline_name]

@exception_logger
def load_segmentation(profile_name: str, database_name: str) -> Dict:
    # Load experiment configuration
    experiment_name = get_profile(profile_name).experiment_config_name
    experiment_config = get_config(experiment_name)
    device = get_device()
    ############ Deep Model ##############
    # Create model instance
    model = load_model(experiment_config)
    if model is not None:
        model.to(device)
    ############ Loss function ##############
    # Create loss instance
    loss = load_loss(experiment_config)
    ############ Optimizer ##############
    # Create optimizer instance
    optimizer = load_optimizer(model, experiment_config)
    # scheduler = load_scheduler(optimizer, experiment_config)
    ############ Comet.ml Experiment ##############
    # Create the experiment instance
    experiment = get_experiment(profile_name=profile_name, dataset=database_name)
    # Logging the model
    experiment.set_model_graph(str(model), overwrite=True)
    # Load profile
    profile = get_profile(profile_name)
    # Load the pipeline
    pipeline_name = profile.pipeline
    if not pipeline_name in experiment_config.pipeline:
        raise ValueError("Pipeline is not supported!")
    ############ Pipeline ##############
    step_func = load_pipeline(pipeline_name)
    ############ Metrics ##############
    # Create metric instances
    metrics = load_metrics(experiment_config, profile.categories)

    to_pil = ToPILImage()

    def __run_pipeline(
        engine: Engine,
        batch,
        step_func: Callable,
        device,
        model: BaseModule,
        loss,
        optimizer: optim.Optimizer,
        metrics: List[BaseMetric],
        experiment: Experiment,
    ) -> Dict:
        profile = get_profile(engine.state.profile_name)
        t = time.time()
        res = step_func(
            engine=engine,
            batch=batch,
            device=device,
            model=model,
            criterion=loss,
            optimizer=optimizer,
            experiment=experiment,
        )

        step_time = time.time() - t
        if profile.enable_logging:
            # Logging loss & step time
            if 'loss' in res:
                engine.state.metrics['loss'] = res['loss']
            engine.state.metrics['step_time'] = step_time
            # Calculate metrics
            if 'metrics' in res:
                engine.state.metrics.update(res['metrics'])
            # Assume Tensor B x C x W x H
            targets = res["y"]
            outputs = res["y_pred"]
            processed = res["y_processed"] if "y_processed" in res else None

            num_samples = res["y"].shape[0]
            for i in range(num_samples):
                out = outputs[i, :, :, :]
                trg = targets[i, :, :, :]
                prc = processed[i, :, :, :] if processed is not None else None

                if profile.enable_image_logging:
                    experiment.log_image(
                        make_tensor_for_comet(out),
                        f"output-{i}",
                        step=engine.state.iteration,
                    )
                    experiment.log_image(
                        make_tensor_for_comet(trg),
                        f"target-{i}",
                        step=engine.state.iteration,
                    )
                    if processed is not None:
                        experiment.log_image(
                            make_tensor_for_comet(prc),
                            f"processed-{i}",
                            step=engine.state.iteration,
                        )

                for m in metrics:
                    m_out = out if prc is None else prc
                    m_out = m_out.squeeze(0) if m_out.shape[0] == 1 else m_out.permute(1,2,0)
                    m_trg = trg.squeeze(0)
                    m_out = m_out.cpu().detach().numpy()
                    m_trg = m_trg.cpu().detach().numpy()
                    m.update((m_out, m_trg))
                    m.compute(engine, experiment)

        return res

    # Initialize the pipeline function
    seg_func = functools.partial(
        __run_pipeline,
        step_func=step_func,
        device=device,
        model=model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        experiment=experiment,
    )
    # Instantiate the engine
    engine = Engine(seg_func)
    # Get Pipeline Configuration
    pipeline_config = experiment_config.pipeline[pipeline_name]
    # Add configurations to the engine state
    engine.state.last_loss = 0
    if experiment_config.pipeline:
        for key, value in pipeline_config.items():
            engine.state_dict_user_keys.append(key)
            setattr(engine.state, key, value)
    engine.state.profile_name = profile_name
    # Save Checkpoint
    run_record = {
        "engine": engine,
        "model": model,
        "optimizer": optimizer,
        "loss": loss,
    }
    enable_checkpoint_save = (
        profile.checkpoint_save if "checkpoint_save" in profile else False
    )
    if enable_checkpoint_save:
        checkpoint_dir = load_settings().checkpoint_dir
        checkpoint_file = f"{pipeline_name}.pt"
        checkpoint_saver = ModelCheckpoint(
            dirname=checkpoint_dir,
            filename_pattern=checkpoint_file,
            filename_prefix="",
            require_empty=False,
            create_dir=True,
            n_saved=1,
            global_step_transform=global_step_from_engine(engine),
        )
        engine.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_saver, run_record)

    # Load Checkpoint
    enable_checkpoint_load = (
        profile.checkpoint_load if "checkpoint_load" in profile else False
    )
    if enable_checkpoint_load:
        checkpoint_dir = load_settings().checkpoint_dir
        checkpoint_file = os.path.join(checkpoint_dir, f"{pipeline_name}.pt")
        if os.path.isfile(checkpoint_file):
            checkpoint_obj = torch.load(checkpoint_file, map_location=get_device())
            ModelCheckpoint.load_objects(to_load=run_record, checkpoint=checkpoint_obj)

    @engine.on(Events.ITERATION_COMPLETED(every=1))
    def __log_training(engine):
        lr = 0
        epoch = engine.state.epoch
        max_epochs = engine.state.max_epochs
        iteration = engine.state.iteration
        step_time = engine.state.step_time if hasattr(engine.state, "step_time") else 0
        print(f"Epoch {epoch}/{max_epochs} [{step_time}] : {iteration} - batch loss: {engine.state.last_loss}, lr: {lr}")

    @engine.on(Events.ITERATION_COMPLETED(every=1))
    def __log_metrics(engine):
        metrics = engine.state.metrics
        experiment.log_metrics(
            metrics,
            step=engine.state.iteration, 
            epoch=engine.state.epoch
        )

    @engine.on(Events.STARTED)
    def __train_process_started(engine):
        experiment.train()
        logging.info("Training is started ...")

    @engine.on(Events.COMPLETED)
    def __train_process_ended(engine):
        logging.info("Training is ended ...")
        experiment.end()

    return run_record

