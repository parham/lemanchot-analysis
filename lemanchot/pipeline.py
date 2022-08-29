
import functools
import logging
from typing import Callable, List, Union
from dotmap import DotMap

from comet_ml import Experiment
from ignite.engine import Engine

import torch.optim as optim
from torch.optim import Optimizer

from ignite.engine.events import Events

from lemanchot.core import BaseCore, exception_logger, get_experiment, get_profile, running_time
from lemanchot.loss.core import BaseLoss, load_loss
from lemanchot.models.core import BaseModule, load_model

def load_optimizer(model : BaseModule, experiment_config : DotMap) -> Optimizer:
    params = model.parameters()
    optim_name = experiment_config.optimizer.name
    optim_config = experiment_config.optimizer.config
    return {
        'SGD' : lambda ps, config: optim.SGD(ps, **config),
        'Adam' : lambda ps, config: optim.Adam(ps, **config)
    }[optim_name](params, optim_config)

__pipeline_handler = {}

def pipeline_register(name : Union[str, List[str]]):
    """Register a pipeline into the pipeline repository

    Args:
        name (Union[str, List[str]]): the pipeline's or list of pipeline's name(s)
    """
    def __embed_func(func):
        global __model_handler
        hname = name if isinstance(name, list) else [name]
        if not issubclass(func, BaseModule):
            raise NotImplementedError('The specified pipeline is not correctly implemented!')

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
def load_pipeline(pipeline_name : str) -> Callable:

    if not pipeline_name in list_pipelines():
        msg = f'{pipeline_name} model is not supported!'
        logging.error(msg)
        raise ValueError(msg)
    
    return __pipeline_handler[pipeline_name]

@exception_logger
def load_segmentation(
    database_name : str,
    categories : List[str],
    experiment_config : DotMap
) -> Engine:
    
    # Check if model configuration is available!
    if 'model' in experiment_config:
        raise ValueError('Model must be defined in the experiment configuration!')
    # Create model instance
    model = load_model(experiment_config)
    # Check if loss configuration is available!
    if 'loss' in experiment_config:
        raise ValueError('Loss must be defined in the experiment configuration!')
    # Create loss instance
    loss = load_loss(experiment_config)
    # Check if optimizer configuration is available!
    if 'optimizer' in experiment_config:
        raise ValueError('Optimizer must be defined in the experiment configuration!')
    # Create optimizer instance
    optimizer = load_optimizer(model, experiment_config)
    # Create the experiment instance
    experiment = get_experiment(dataset=database_name)
    # Logging the model
    experiment.set_model_graph(str(model), overwrite=True)
    # Load the pipeline
    pipeline_name = get_profile().pipeline
    if not pipeline_name in experiment_config.pipeline:
        raise ValueError('Pipeline is not supported!')

    step_func = load_pipeline(pipeline_name)

    seg_func = functools.partial(step_func,
        model=model,
        loss=loss,
        optimizer=optimizer,
        experiment=experiment
    )

    engine = Engine(seg_func)

    @engine.on(Events.ITERATION_COMPLETED(every=100))
    def log_training(engine):
        batch_loss = engine.state.output
        lr = optimizer.param_groups[0]['lr']
        epoch = engine.state.epoch
        max_epochs = engine.state.max_epochs
        iteration = engine.state.iteration
        print(f"Epoch {epoch}/{max_epochs} : {iteration} - batch loss: {batch_loss}, lr: {lr}")

    return engine

@pipeline_register("simple_trainer")
def simple_train_step__(
    engine : Engine, 
    batch,
    model : BaseModule,
    loss : BaseLoss,
    optimizer : Optimizer,
    experiment : Experiment
):
    inputs, targets = batch
    loss.prepare_loss(ref=batch[0])
    model.train()
    optimizer.zero_grad()
    outputs = model(inputs)
    loss_value = loss(outputs, targets)
    loss_value.backward()
    optimizer.step()
    return loss_value.item()