import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from a2c_ppo_acktr.model import DMPNet
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian, DiagGaussianDist
from a2c_ppo_acktr.utils import init
from dmp.utils.dmp_layer import DMPIntegrator, DMPParameters
from a2c_ppo_acktr import pytorch_util as ptu
import cv2
from dmp.utils.smnist_loader import MatLoader
# from os.path import dirname, realpath
import os
import sys
import argparse
from datetime import datetime
from dmp.utils.mnist_cnn import CNN