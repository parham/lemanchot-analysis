

"""
    @project LeManchot-Analysis : Core components
    @organization Laval University
    @lab MiViM Lab
    @supervisor Professor Xavier Maldague
    @industrial-partner TORNGATS
"""

from abc import ABC, abstractmethod
from typing import Dict
from ignite.engine import Engine

class BaseSaver(ABC):
    def __init__(self,
        root_dir : str,
        pipeline_name : str
    ) -> None:
        self.root_dir = root_dir
        self.pipeline_name = pipeline_name
    
    @abstractmethod
    def __call__(self, engine: Engine, to_save: Dict):
        pass

class ImageSaver(BaseSaver):
    def __init__(self, root_dir: str, pipeline_name: str) -> None:
        super().__init__(root_dir, pipeline_name)