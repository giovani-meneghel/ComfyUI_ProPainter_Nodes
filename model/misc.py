import os
import re
import random
import time
import torch
from torch import nn
import numpy as np
from os import path as osp


def constant_init(module, val, bias=0):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)




IS_HIGH_VERSION = [
    int(m)
    for m in list(
        re.findall(
            r"^([0-9]+)\.([0-9]+)\.([0-9]+)([^0-9][a-zA-Z0-9]*)?(\+git.*)?$",
            torch.__version__,
        )[0][:3]
    )
] >= [1, 12, 0]


def gpu_is_available():
    if IS_HIGH_VERSION:
        if torch.backends.mps.is_available():
            return True
    return (
        True
        if torch.cuda.is_available() and torch.backends.cudnn.is_available()
        else False
    )


def get_device(gpu_id=None):
    if gpu_id is None:
        gpu_str = ""
    elif isinstance(gpu_id, int):
        gpu_str = f":{gpu_id}"
    else:
        raise TypeError("Input should be int value.")

    if IS_HIGH_VERSION:
        if torch.backends.mps.is_available():
            return torch.device("mps" + gpu_str)
    return torch.device(
        "cuda" + gpu_str
        if torch.cuda.is_available() and torch.backends.cudnn.is_available()
        else "cpu"
    )


def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_time_str():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.

    Returns:
        A generator for all the interested files with relative pathes.
    """
    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith(".") and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if suffix is None or return_path.endswith(suffix):
                    yield return_path
            elif recursive:
                yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
            else:
                continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)
