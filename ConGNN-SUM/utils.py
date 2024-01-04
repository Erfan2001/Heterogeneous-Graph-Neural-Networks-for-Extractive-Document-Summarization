import torch
from tools.logger import *


def set_device(hps):
    if hps.cuda and hps.gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
        logger.info("[INFO] Use cuda")
    else:
        device = torch.device("cpu")
        logger.info("[INFO] Use CPU")
    hps.device = device
    return hps
