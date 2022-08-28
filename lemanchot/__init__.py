
""" 
    @title        Multi-modal texture analysis to enhance drone-based thermographic inspection of structures 
    @organization Laval University
    @partner      TORNGATS
"""

from .core import *

# Check CUDA Availability
import torch, gc

if not torch.cuda.is_available():
    logging.warning('CUDA is not available')
else:
    # Clearing the GPU memory
    gc.collect()
    torch.cuda.empty_cache()

# Initialize the folders
from pathlib import Path
Path("./logs").mkdir(parents=True, exist_ok=True)
Path("./datasets").mkdir(parents=True, exist_ok=True)

# Initialize the logging
initialize_log()