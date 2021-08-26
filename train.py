import ruamel.yaml as yaml
from argparse import Namespace
import gym
import matplotlib.pyplot as plt
from math import tanh
import numpy as np
import os
import PIL
import threading
from time import time
from tqdm import tqdm
import torch
from torch.optim import Adam
from torch.utils import data
import torchvision
import warnings

warnings.filterwarnings(action='ignore')

from models import *
from utils import *
from utils.envs import Atari


def main():
    with open('config.yaml', 'r') as f:
        config = yaml.load(f)
        config = Namespace(**config['atari_pong'])

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    if torch.cuda.is_available() and config.device:
        device = torch.device('cuda')
        torch.cuda.manual_seed(config.seed)
    else:
        device = torch.device('cpu')

    print('device :', device)
    print('env_name : ', config.env_name)

    env = Atari(config.env_name, config.action_repeat, config.image_size, config.grayscale,
                life_done=False, sticky_actions=True, all_actions=True)

    env = gym.make(config.env_name, frameskip=config.action_repeat)
    config.action_size = env.action_space.n
    config.observation_shape = env.observation_space.shape

    wm = WorldModel(config)


if __name__ == '__main__':
    main()
