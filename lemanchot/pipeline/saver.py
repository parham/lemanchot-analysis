

"""
    @project LeManchot-Analysis : Core components
    @organization Laval University
    @lab MiViM Lab
    @supervisor Professor Xavier Maldague
    @industrial-partner TORNGATS
"""

from abc import abstractmethod
import os
from pathlib import Path
from typing import Dict
from comet_ml import Experiment
from PIL import Image

from ignite.engine import Engine
from ignite.handlers import ModelCheckpoint

class ModelLogger_CometML:
    """ ModelLogger_CometML provides the feature to log models in comet.ml """
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
        """Logging the model each time, it calls

        Args:
            engine (Engine): the engine handler.
            to_save (Dict): the dictionary containing the model.
        """
        checkpoint_fpath = self.checkpoint_handler.last_checkpoint
        self.experiment.log_model(
            name=f'{self.pipeline_name}-{self.model_name}', # ::{engine.state.epoch}
            file_or_folder=str(checkpoint_fpath),
            metadata=engine.state.metrics,
            overwrite=True
        )


class ImageSaver:
    def __init__(self, root_dir) -> None:
        self.root_dir = root_dir
        if not os.path.isdir(root_dir):
            Path(root_dir).mkdir(parents=True, exist_ok=True)

    def __call__(self, engine: Engine, label : str, img):
        file = os.path.join(self.root_dir, f'{label}.png')
        Image.fromarray(img).save(file)