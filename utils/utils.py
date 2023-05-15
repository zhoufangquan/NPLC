from operator import is_
import os
import os
import time
import random
import logging
import numpy as np
from pathlib import Path

import torch


def seed_everything(seed=2023):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


logger = logging.getLogger()


def init_logger(log_file=None, log_file_level=logging.NOTSET):
    '''
    Example:
        >>> init_logger(log_file)
        >>> logger.info("abc'")
    '''
    if isinstance(log_file, Path):
        log_file = str(log_file)
    log_format = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                                   datefmt='%m/%d/%Y %H:%M:%S')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]
    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        # file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger


def save_last_model(path, model, epoch, is_train=False):
    if is_train:
        out = os.path.join(path, "checkpoint_last_train.tar")
    else:
        out = os.path.join(path, "checkpoint_last_boost.tar")
    state = {'net': model.state_dict(), 'epoch': epoch}
    torch.save(state, out)

def save_best_model(path, model, is_train=False):
    if is_train:
        out = os.path.join(path, "checkpoint_best_train.tar")
    else:
        out = os.path.join(path, "checkpoint_best_boost.tar")
    state = {'net': model.state_dict(), 'epoch': 0}
    torch.save(state, out)

def save_model(path, model, type):
    out = os.path.join(path, f"checkpoint_last_{type}.tar")
    state = {'net': model.state_dict(), 'epoch': 0}
    torch.save(state, out)
