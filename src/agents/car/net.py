import numpy as np
import torch
import torch.nn.functional as F

from typing import Dict
from .backbone import Backbone

class NeuralNetAgent(torch.nn.Module):

    def __init__(self, args):
        """
        Args:
            obs_size: observation/state size of the environment
            n_actions: number of discrete actions available in the environment
            hidden_size: size of hidden layers
        """
        super(NeuralNetAgent, self).__init__()
        self.backbone = Backbone.from_argparse_args(args)

    @staticmethod
    def add_argparse_args(parser):
        Backbone.add_argparse_args(parser)
        return parser

    @staticmethod
    def from_argparse_args(args):
        return NeuralNetAgent(args)

    def forward(self, sample: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.backbone.forward(sample)

    def reset(self, env):
        self.goal_state = env.goal_state
        self.obstacle_centers = env.obstacle_centers
        self.obstacle_radii = env.obstacle_radii

    def get_action(self, state: np.ndarray) -> np.ndarray:

        relative_goal = state[:3] - self.goal_state[:3]
        relative_obstacle_centers = state[:2] - self.obstacle_centers
        trunc_state = state[3:]

        inputs = {
            'trunc_state': torch.from_numpy(trunc_state.astype(np.float32)),
            'state': torch.from_numpy(state.astype(np.float32)),
            'relative_goal': torch.from_numpy(relative_goal.astype(np.float32)),
            'obstacle_centers': torch.from_numpy(relative_obstacle_centers.astype(np.float32)),
            'obstacle_radii': torch.from_numpy(self.obstacle_radii.astype(np.float32))
        }
        return self.backbone(inputs).detach().clone().numpy()
