"""
    @project LeManchot-Analysis : Core components
    @organization Laval University
    @lab MiViM Lab
    @supervisor Professor Xavier Maldague
    @industrial-partner TORNGATS
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Union
from uuid import uuid4

import numpy as np
import torch
import torch.optim as optim
from dotmap import DotMap
from ignite.engine import Engine
from ignite.engine.events import Events
from ignite.handlers import ModelCheckpoint, global_step_from_engine
from ignite.handlers.param_scheduler import (
    CosineAnnealingScheduler,
    LRScheduler,
    ReduceLROnPlateauScheduler,
)
from lemanchot.core import (
    exception_logger,
    get_config,
    get_device,
    get_experiment,
    get_or_default,
    get_profile,
    load_settings,
)
from lemanchot.loss import load_loss
from lemanchot.metrics import load_metrics
from lemanchot.models import BaseModule, load_model
from lemanchot.pipeline.saver import ImageSaver, ModelLogger_CometML
from lemanchot.pipeline.wrapper import load_wrapper
from lemanchot.visualization import COLORS
from torch.optim.lr_scheduler import ExponentialLR, StepLR


def load_optimizer(model: BaseModule, experiment_config: DotMap) -> optim.Optimizer:
    """Load the optimizer based on given configuration

    Args:
        model (BaseModule): the model
        experiment_config (DotMap): configuration for optimizer

    Returns:
        optim.Optimizer: the instantiated optimizer
    """

    if model is None or not "optimizer" in experiment_config:
        return None

    params = model.parameters()

    optim_name = experiment_config.optimizer.name
    optim_config = (
        experiment_config.optimizer.config
        if "config" in experiment_config.optimizer
        else {}
    )

    return {
        "SGD": lambda ps, config: optim.SGD(ps, **config),
        "Adam": lambda ps, config: optim.Adam(ps, **config),
        "Adadelta": lambda ps, config: optim.Adadelta(ps, **config),
        "AdamW": lambda ps, config: optim.AdamW(ps, **config),
        "Adamax": lambda ps, config: optim.Adamax(ps, **config),
        "RMSprop": lambda ps, config: optim.RMSprop(ps, **config),
    }[optim_name](params, optim_config)


def load_scheduler(
    engine: Engine, optimizer: optim.Optimizer, experiment_config: DotMap
):
    """Load Optimizer Scheduler based on the given configuration

    Args:
        engine (Engine): the engine handler
        optimizer (optim.Optimizer): the optimizer handler
        experiment_config (DotMap): the given configuration
    """
    if optimizer is None or not "scheduler" in experiment_config:
        return None

    scheduler_name = experiment_config.scheduler.name
    scheduler_config = (
        experiment_config.scheduler.config
        if "config" in experiment_config.scheduler
        else {}
    )

    def __step_lr(engine: Engine, optimizer: optim.Optimizer, scheduler_config: DotMap):
        # Sample configuration sample
        # "scheduler" : {
        #     "name" : "StepLR",
        #     "config" : {
        #         "step_size" : 3,
        #         "gamma": 0.1
        #     }
        # }
        steplr = StepLR(
            optimizer,
            step_size=scheduler_config["step_size"],
            gamma=scheduler_config["gamma"],
        )
        scheduler = LRScheduler(steplr)

        period = (
            scheduler_config["period"] if "period" in scheduler_config else "iteration"
        )
        if period == "iteration":
            engine.add_event_handler(Events.ITERATION_COMPLETED, scheduler)
        elif period == "epoch":
            engine.add_event_handler(Events.EPOCH_COMPLETED, scheduler)
        else:
            raise ValueError("The period for scheduler is not supported!")

        @engine.on(Events.ITERATION_COMPLETED)
        def loging_metrics_lr():
            engine.state.metrics["lr"] = optimizer.param_groups[0]["lr"]

        return scheduler

    def __reduce_lr_plateau(
        engine: Engine, optimizer: optim.Optimizer, scheduler_config: DotMap
    ):
        # "scheduler" : {
        #     "name" : "ReduceLROnPlateau",
        #     "config" : {
        #         "metric_name" : "loss",
        #         "save_history" : true,
        #         "mode" : "min",
        #         "factor" : 0.5,
        #         "patience" : 3,
        #         "threshold_mode" : "rel",
        #         "threshold" : 0.1
        #     }
        # }
        scheduler = ReduceLROnPlateauScheduler(
            optimizer,
            metric_name=scheduler_config["metric_name"],
            save_history=scheduler_config["save_history"],
            mode=scheduler_config["mode"],
            factor=scheduler_config["factor"],
            patience=scheduler_config["patience"],
            threshold_mode=scheduler_config["threshold_mode"],
            threshold=scheduler_config["threshold"],
            trainer=engine,
        )
        period = (
            scheduler_config["period"] if "period" in scheduler_config else "iteration"
        )
        if period == "iteration":
            engine.add_event_handler(Events.ITERATION_COMPLETED, scheduler)
        elif period == "epoch":
            engine.add_event_handler(Events.EPOCH_COMPLETED, scheduler)
        else:
            raise ValueError("The period for scheduler is not supported!")

        @engine.on(Events.ITERATION_COMPLETED)
        def loging_metrics_lr():
            engine.state.metrics["lr"] = np.mean(
                np.array(engine.state.param_history["lr"][-1])
            )

        return scheduler

    def __consine_annealing_lr(
        engine: Engine, optimizer: optim.Optimizer, scheduler_config: DotMap
    ):
        # "scheduler" : {
        #     "name" : "CosineAnnealingLR",
        #     "config" : {
        #         "T_0": 5,
        #         "T_mul" : 1,
        #         "eta_min" : 0.00001
        #     }
        # }
        scheduler = CosineAnnealingScheduler(
            optimizer=optimizer,
            param_name="lr",
            start_value=scheduler_config["start_value"],
            end_value=scheduler_config["end_value"],
            cycle_size=scheduler_config["cycle_size"],
            cycle_mult=scheduler_config["cycle_mult"],
            start_value_mult=scheduler_config["mult"],
            end_value_mult=scheduler_config["mult"],
        )

        # scheduler = LRScheduler(cosine_lr)
        period = (
            scheduler_config["period"] if "period" in scheduler_config else "iteration"
        )
        if period == "iteration":
            engine.add_event_handler(Events.ITERATION_COMPLETED, scheduler)
        elif period == "epoch":
            engine.add_event_handler(Events.EPOCH_COMPLETED, scheduler)
        else:
            raise ValueError("The period for scheduler is not supported!")

        @engine.on(Events.ITERATION_COMPLETED)
        def loging_metrics_lr():
            engine.state.metrics["lr"] = optimizer.param_groups[0]["lr"]

        return scheduler

    def __exponential_lr(
        engine: Engine, optimizer: optim.Optimizer, scheduler_config: DotMap
    ):
        # "scheduler" : {
        #     "name" : "ExponentialLR",
        #     "config" : {
        #         "gamma" : 0.98
        #     }
        # }
        exp_lr = ExponentialLR(optimizer=optimizer, gamma=scheduler_config["gamma"])

        scheduler = LRScheduler(exp_lr)
        period = (
            scheduler_config["period"] if "period" in scheduler_config else "iteration"
        )
        if period == "iteration":
            engine.add_event_handler(Events.ITERATION_COMPLETED, scheduler)
        elif period == "epoch":
            engine.add_event_handler(Events.EPOCH_COMPLETED, scheduler)
        else:
            raise ValueError("The period for scheduler is not supported!")

        @engine.on(Events.ITERATION_COMPLETED)
        def loging_metrics_lr():
            engine.state.metrics["lr"] = optimizer.param_groups[0]["lr"]

        return scheduler

    scheduler = {
        "StepLR": __step_lr,
        "ReduceLROnPlateau": __reduce_lr_plateau,
        "CosineAnnealingLR": __consine_annealing_lr,
        "ExponentialLR": __exponential_lr,
    }[scheduler_name](engine, optimizer, scheduler_config)

    return scheduler


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
    """Initialize and instantiate the pipeline

    Args:
        profile_name (str): the name of the selected profile
        database_name (str): the name of the database

    Returns:
        Dict: the dictionary containing the engine, model, scheduler, and other components.
    """
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
    # Create the image logger
    img_saver = None
    if "image_saving" in profile:
        image_saving = profile.image_saving
        img_saver = ImageSaver(**image_saving)

    wrapper_name = experiment_config.wrapper.name
    seg_func = load_wrapper(
        wrapper_name=wrapper_name,
        step_func=step_func,
        device=device,
        model=model,
        loss=loss,
        optimizer=optimizer,
        metrics=metrics,
        experiment=experiment,
        img_saver=img_saver,
    )

    # Log hyperparameters
    experiment.log_parameters(experiment_config.toDict())
    # Log colormap-encoding
    cnames = COLORS.names()
    colors = {
        cls_name: cnames[color_idx]
        for cls_name, color_idx in get_profile(profile_name).categories.items()
    }
    experiment.log_parameters({"colors": colors})
    # Instantiate the engine
    engine = Engine(seg_func)
    # Create scheduler instance
    scheduler = load_scheduler(engine, optimizer, experiment_config)
    # Get Pipeline Configuration
    pipeline_config = experiment_config.pipeline[pipeline_name]
    # Add configurations to the engine state
    engine.state.last_loss = 0
    if experiment_config.pipeline:
        for key, value in pipeline_config.items():
            engine.state_dict_user_keys.append(key)
            setattr(engine.state, key, value)
    engine.state.profile_name = profile_name

    run_record = {
        "engine": engine,
        "model": model,
        "optimizer": optimizer,
        "loss": loss,
    }
    enable_checkpoint_save = get_or_default(profile, "checkpoint_save", False)
    enable_checkpoint_log = get_or_default(profile, "checkpoint_log_cometml", False)

    checkpoint_file = f"{pipeline_name}-{model.name}-{str(uuid4())[0:8]}.pt"
    if enable_checkpoint_save:
        experiment.log_parameter(name="checkpoint_file", value=checkpoint_file)
        checkpoint_dir = load_settings().checkpoint_dir
        checkpoint_file = checkpoint_file
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
        # Logging Model
        if enable_checkpoint_log:
            checkpoint_logger = ModelLogger_CometML(
                pipeline_name, model.name, experiment, checkpoint_saver
            )
            engine.add_event_handler(
                Events.EPOCH_COMPLETED, checkpoint_logger, run_record
            )

    # Load Checkpoint
    enable_checkpoint_load = get_or_default(profile, "checkpoint_load", False)
    if enable_checkpoint_load:
        checkpoint_dir = load_settings().checkpoint_dir
        checkpoint_file = os.path.join(checkpoint_dir, f"{profile.checkpoint_file}")
        if os.path.isfile(checkpoint_file):
            checkpoint_obj = torch.load(checkpoint_file, map_location=get_device())
            if profile.load_weights_only:
                run_record["model"].load_state_dict(checkpoint_file["model"])
            else:
                ModelCheckpoint.load_objects(
                    to_load=run_record, checkpoint=checkpoint_obj
                )

    @engine.on(Events.ITERATION_COMPLETED(every=1))
    def __log_training(engine):
        lr = engine.state.metrics["lr"] if "lr" in engine.state.metrics else 0
        epoch = engine.state.epoch
        max_epochs = engine.state.max_epochs
        iteration = engine.state.iteration
        step_time = engine.state.step_time if hasattr(engine.state, "step_time") else 0
        vloss = get_or_default(engine.state.metrics, "loss", 0)
        print(
            f"Epoch {epoch}/{max_epochs} [{step_time}] : {iteration} - batch loss: {vloss:.4f}, lr: {lr:.4f}"
        )

    @engine.on(Events.ITERATION_COMPLETED(every=1))
    def __log_metrics(engine):
        profile = get_profile(engine.state.profile_name)
        if profile.enable_logging:
            metrics = engine.state.metrics
            experiment.log_metrics(
                dict(metrics), step=engine.state.iteration, epoch=engine.state.epoch
            )

    @engine.on(Events.STARTED)
    def __train_process_started(engine):
        experiment.train()
        logging.info("Training started ...")

    @engine.on(Events.COMPLETED)
    def __train_process_ended(engine):
        logging.info("Training ended ...")
        experiment.end()

    return run_record
