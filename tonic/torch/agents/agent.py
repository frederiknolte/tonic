import os
import random

import numpy as np
import torch

from tonic import agents, logger  # noqa


class Agent(agents.Agent):
    def initialize(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)

    def save(self, path):
        path = path + '.pt'
        logger.log(f'Saving weights to {path}')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        path = path + '.pt'
        logger.log(f'Loading weights from {path}')
        self.model.load_state_dict(torch.load(path))

    def to(self, device):
        self.model = self.model.to(device)
        self.replay.set_device(device)
