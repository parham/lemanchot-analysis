
""" 
    @title        Multi-modal texture analysis to enhance drone-based thermographic inspection of structures 
    @organization Laval University
    @partner      TORNGATS
"""

import json
import os
import random
import string
import threading
import functools

import logging
import logging.config

from time import time
from dotmap import DotMap
from typing import Any, Dict
from datetime import datetime

import torch
import yaml

from comet_ml import Experiment

"""Generate random strings

Returns:
    str: random string
"""
generate_random_str = lambda x: ''.join(random.choice(string.ascii_lowercase) for i in range(x))

class BaseCore(torch.nn.Module):
    """Base class for all module like components"""
    def __init__(self, name : str, device : str, config : Dict[str, Any]) -> None:
        super().__init__()
        # Initialize the configuration
        for key, value in config.items():
            setattr(self, key, value)
        self.name = name
        self.device = torch.device(device)
        self.to(device)

def initialize_log():
    """Initialize the log configuration"""
    def _init_impl(log_cfile):
        if os.path.isfile(log_cfile):
            with open(log_cfile, 'r') as f:
                config = yaml.safe_load(f.read())
                logging.config.dictConfig(config)
                logging.getLogger().setLevel(logging.INFO)
            logging.info(
                'Logging is configured based on the defined configuration file.')
        else:
            logging.error('the logging configuration file does not exist')

    _init_impl('log_config.yml')

def exception_logger(function):
    """
    A decorator that wraps the passed in function and logs 
    exceptions should one occur
    """
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except:
            # log the exception
            err = "There was an exception in  " + function.__name__
            logging.exception(err)
            # re-raise the exception
            raise
    return wrapper

def running_time(func):
    """ Calculate the runtime of decorated function """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t = time.time()
        res = func(*args, **kwargs)
        runtime = time.time() - t
        res = res + (runtime, ) if isinstance(res,tuple) or \
            isinstance(res,list) else (res, runtime)
        return res
    return wrapper

def synchronized(func):
    """ Make a function synchronized (thread-safe) """
    lock = threading.Lock()
    @functools.wraps(func)
    def _wrap(*args, **kwargs):
        logging.info("Calling '%s' with Lock %s" % (func.__name__, id(lock)))
        with lock:
            return func(*args, **kwargs)
    return _wrap

def read_config(config_file : str) -> DotMap:
    """Read the configuration file

    Args:
        config_file (str): the file path of the configuration

    Returns:
        DotMap: the configuration
    """
    config = dict()
    with open(config_file, 'r') as cfile:
        config = json.load(cfile)
    return DotMap(config)

# The path of the configuration folder
__LEMANCHOT_VT_CONFIG_DIR__ = 'LEMANCHOT_VT_CONFIG_DIR'

@functools.lru_cache(maxsize=5)
def get_config(config_name : str) -> DotMap:
    """Get the selected configuration

    Args:
        config_name (str): configuration name

    Returns:
        DotMap: configuration
    """
    cdir = os.environ.get(__LEMANCHOT_VT_CONFIG_DIR__) if __LEMANCHOT_VT_CONFIG_DIR__ in os.environ else './configs'
    cfile = os.path.join(cdir,config_name + '.json')
    if os.path.isfile(cfile):
        raise ValueError(f'{cfile} does not exist!')
    return read_config(cfile)

# The environment variable for system setting
__LEMANCHOT_VT_SETTING_PATH__ = 'LEMANCHOT_VT_SETTING_PATH'
# The instance of the configuration
__setting_instance = None

@exception_logger
def load_settings() -> DotMap:
    """Load the configuration (*.json)

    Returns:
        DotMap: system settings
    """
    @synchronized
    def __synchronized_read_config():
        cpath = os.environ.get(__LEMANCHOT_VT_SETTING_PATH__) if __LEMANCHOT_VT_SETTING_PATH__ in os.environ else './settings.json'
        return read_config(cpath)

    global __setting_instance
    if __setting_instance is None:
        __setting_instance =  __synchronized_read_config()
    return __setting_instance

def get_device() -> str:
    """Get selected device based on system settings

    Returns:
        str: device name
    """
    settings = load_settings()
    return settings.device

# The selected profile
__LEMANCHOT_VT_PROFILE__ = 'LEMANCHOT_VT_PROFILE'

@functools.lru_cache(maxsize=1)
def get_profile_name() -> str:
    """Extracting profile name

    Raises:
        ValueError: if the environment variable does not exist

    Returns:
        str: profile name
    """
    if not __LEMANCHOT_VT_SETTING_PATH__ in os.environ:
        raise ValueError('LEMANCHOT_VT_PROFILE is not defined!')
    return os.environ.get(__LEMANCHOT_VT_SETTING_PATH__)

@functools.lru_cache(maxsize=1)
def get_profile() -> DotMap:
    """Get the selected profile

    Raises:
        ValueError: if the profile does not exist!

    Returns:
        DotMap: given profile
    """
    profile_name = get_profile_name()
    settings = load_settings()
    if not profile_name in settings.profiles:
        raise ValueError(f'{profile_name} is not defined!')
    profile = settings.profiles[profile_name]
    profile['name'] = profile_name
    return profile

# The instance of the experiment
__experiment_instance = {}

def get_experiment(dataset : str = None) -> Experiment:
    """Get the experiment

    Args:
        dataset (str): dataset name

    Returns:
        Experiment: the created experiment
    """
    global __experiment_instance

    @synchronized
    def __create_experiment():
        tnow = datetime.now()
        profile = get_profile()
        __experiment_instance = Experiment(
            api_key=profile.api_key,
            project_name=profile.project_name,
            workspace=profile.workspace,
            log_git_metadata=profile.log_git_metadata,
            log_env_gpu=profile.log_env_gpu,
            log_env_cpu=profile.log_env_cpu,
            log_env_host=profile.log_env_host,
            disabled=profile.enable_loggings
        )
        __experiment_instance.set_name('%s_%s_%s' % (profile['name'], tnow.strftime('%Y%m%d-%H%M'), dataset))
        __experiment_instance.add_tag(dataset)

    if __experiment_instance is None:
        __experiment_instance = __create_experiment()

    return __experiment_instance

