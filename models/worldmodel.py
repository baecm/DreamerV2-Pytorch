import torch
import torch.nn as nn

from utils.buffer import TransitionBuffer


class WorldModel(nn.Module):
    def __init__(self, config):
        super(WorldModel, self).__init__()
        self.buffer = TransitionBuffer(config.capacity, config.observation_shape, config.action_size, config.sequence_length, config.batch_size)
        self.rssm
        self.actor

