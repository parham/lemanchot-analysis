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
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ExponentialLR
from torchvision.transforms import ToPILImage

from ignite.engine import Engine
from ignite.engine.events import Events
from ignite.handlers import ModelCheckpoint, global_step_from_engine
from ignite.handlers.param_scheduler import (
    LRScheduler, 
    ReduceLROnPlateauScheduler
)

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
        not 'optimizer' in experiment_config:
        return None

    params = model.parameters()

    optim_name = experiment_config.optimizer.name
    optim_config = (
        experiment_config.optimizer.config
        if 'config' in experiment_config.optimizer else {}
    )

    return {
        'SGD' : lambda ps, config: optim.SGD(ps, **config),
        'Adam' : lambda ps, config: optim.Adam(ps, **config),
    }[optim_name](params, optim_config)

def load_scheduler(
    engine : Engine,
    optimizer: optim.Optimizer, 
    experiment_config: DotMap
):
    if optimizer is None or \
        not 'scheduler' in experiment_config:
        return None

    scheduler_name = experiment_config.scheduler.name
    scheduler_config = (
        experiment_config.scheduler.config
        if 'config' in experiment_config.scheduler else {}
    )

    def __step_lr(
        engine : Engine, 
        optimizer: optim.Optimizer, 
        scheduler_config: DotMap
    ):
        # Sample configuration sample
        # "scheduler" : {
        #     "name" : "StepLR",
        #     "config" : {
        #         "step_size" : 3,
        #         "gamma": 0.1
        #     }
        # }
        steplr = StepLR(optimizer, 
            step_size=scheduler_config['step_size'], 
            gamma=scheduler_config['gamma']
        )
        scheduler = LRScheduler(steplr)
        engine.add_event_handler(Events.ITERATION_STARTED, scheduler)

        @engine.on(Events.ITERATION_COMPLETED)
        def loging_metrics_lr():
            engine.state.metrics['lr'] = optimizer.param_groups[0]['lr']
        
        return scheduler

    def __reduce_lr_plateau(
        engine : Engine, 
        optimizer: optim.Optimizer, 
        scheduler_config: DotMap
    ):
        # "scheduler" : {
        #     "name" : "ReduceOnPlateau",
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
            metric_name=scheduler_config['metric'],
            save_history=scheduler_config['save_history'], 
            mode=scheduler_config['mode'], 
            factor=scheduler_config['factor'], 
            patience=scheduler_config['patience'],
            threshold_mode=scheduler_config['threshold_mode'],
            threshold=scheduler_config['threshold'],
            trainer=engine
        )
        engine.add_event_handler(Events.ITERATION_STARTED, scheduler)

        @engine.on(Events.ITERATION_COMPLETED)
        def loging_metrics_lr():
            engine.state.metrics['lr'] = engine.state.param_history["lr"]

        return scheduler

    def __consine_annealing_lr(
        engine : Engine, 
        optimizer: optim.Optimizer, 
        scheduler_config: DotMap
    ):
        # "scheduler" : {
        #     "name" : "CosineAnnealingLR",
        #     "config" : {
        #         "T_max" : 10,
        #         "eta_min" : 0.01
        #     }
        # }
        cosine_lr = CosineAnnealingLR(
            optimizer=optimizer, 
            T_max=scheduler_config['T_max'], 
            eta_min=scheduler_config['eta_min']
        )

        scheduler = LRScheduler(cosine_lr)
        engine.add_event_handler(Events.ITERATION_STARTED, scheduler)

        @engine.on(Events.ITERATION_COMPLETED)
        def loging_metrics_lr():
            engine.state.metrics['lr'] = optimizer.param_groups[0]['lr']
        
        return scheduler

    def __exponential_lr(
        engine : Engine, 
        optimizer: optim.Optimizer, 
        scheduler_config: DotMap
    ):
        # "scheduler" : {
        #     "name" : "ExponentialLR",
        #     "config" : {
        #         "gamma" : 0.98
        #     }
        # }
        exp_lr = ExponentialLR(
            optimizer=optimizer, 
            gamma=scheduler_config['gamma']
        )

        scheduler = LRScheduler(exp_lr)
        engine.add_event_handler(Events.ITERATION_STARTED, scheduler)

        @engine.on(Events.ITERATION_COMPLETED)
        def loging_metrics_lr():
            engine.state.metrics['lr'] = optimizer.param_groups[0]['lr']
        
        return scheduler

    scheduler = {
        'StepLR' : __step_lr,
        'ReduceLROnPlateau' : __reduce_lr_plateau,
        'CosineAnnealingLR' : __consine_annealing_lr,
        'ExponentialLR' : __exponential_lr
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
            # Logging imagery results
            for key, img in res.items():
                if not key.startswith('y_'):
                    continue

                num_samples = img.shape[0]
                for i in range(num_samples):
                    sample = img[i, :, :, :]
                    if profile.enable_image_logging:
                        experiment.log_image(
                            make_tensor_for_comet(sample),
                            f"output-{i}",
                            step=engine.state.iteration,
                        )

            targets = res["y"]
            outputs = res["y_pred"] if not "y_processed" in res else res["y_processed"]

            num_samples = targets.shape[0]
            for i in range(num_samples):
                out = outputs[i, :, :, :]
                trg = targets[i, :, :, :]

                out = (out.squeeze(0) if out.shape[0] == 1 else out.permute(1,2,0)).cpu().detach().numpy()
                trg = trg.squeeze(0).cpu().detach().numpy()
                for m in metrics:
                    m.update((out, trg))
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
    # Save Checkpoint
    run_record = {
        "engine": engine,
        "model": model,
        "optimizer": optimizer,
        "scheduler" : scheduler,
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
        lr = engine.state.metrics['lr'] if 'lr' in engine.state.metrics else 0
        epoch = engine.state.epoch
        max_epochs = engine.state.max_epochs
        iteration = engine.state.iteration
        step_time = engine.state.step_time if hasattr(engine.state, "step_time") else 0
        print(f"Epoch {epoch}/{max_epochs} [{step_time}] : {iteration} - batch loss: {engine.state.last_loss}, lr: {lr}")

    @engine.on(Events.ITERATION_COMPLETED(every=1))
    def __log_metrics(engine):
        profile = get_profile(engine.state.profile_name)
        if profile.enable_logging:
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

