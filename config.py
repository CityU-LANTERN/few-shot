import os

# PATH = os.path.dirname(os.path.realpath(__file__))
# DATA_PATH = None
pltf = 'linux'    # {win, linux}
if pltf == 'win':
    PATH = "//172.18.36.77/datasets/few-shot-experiments"   # for windows
    DATA_PATH = "//172.18.36.77/datasets/datasets"          # for windows
else:
    PATH = "/liaoweiduo/few-shot-experiments"               # for ubuntu server
    DATA_PATH = "/liaoweiduo/datasets"                      # for ubuntu server


EPSILON = 1e-8

if DATA_PATH is None:
    raise Exception('Configure your data folder location in config.py before continuing!')
