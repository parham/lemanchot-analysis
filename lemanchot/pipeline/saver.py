

"""
    @project LeManchot-Analysis : Core components
    @organization Laval University
    @lab MiViM Lab
    @supervisor Professor Xavier Maldague
    @industrial-partner TORNGATS
"""

from abc import ABC, abstractmethod
from typing import Dict

from comet_ml import Experiment

from ignite.engine import Engine
from ignite.handlers import ModelCheckpoint

class ModelLogger_CometML:
    def __init__(self,
        pipeline_name : str,
        model_name : str,
        experiment : Experiment,
        checkpoint_handler : ModelCheckpoint
    ) -> None:
        super().__init__()
        self.pipeline_name = pipeline_name
        self.model_name = model_name
        self.experiment = experiment
        self.checkpoint_handler = checkpoint_handler

    def __call__(self, engine: Engine, to_save: Dict):
        checkpoint_fpath = self.checkpoint_handler.last_checkpoint
        self.experiment.log_model(
            name=f'{self.pipeline_name}-{self.model_name}',
            file_or_folder=str(checkpoint_fpath),
            metadata=engine.state.metrics,
            overwrite=True
        )