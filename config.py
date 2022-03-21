import os


# PATH = os.path.dirname(os.path.realpath(__file__))
PATH = "//10.20.2.245/datasets/few-shot-experiments"            # for windows
# PATH = "/liaoweiduo/few-shot-experiments"                     # for ubuntu server

DATA_PATH = "//10.20.2.245/datasets/datasets"       # for windows
# DATA_PATH = "/liaoweiduo/datasets"                # for ubuntu server
# DATA_PATH = None

EPSILON = 1e-8

if DATA_PATH is None:
    raise Exception('Configure your data folder location in config.py before continuing!')
