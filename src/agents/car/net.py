import numpy as np
import torch
import torch.nn.functional as F

from typing import Dict
from .backbone import Backbone

class NeuralNetAgent(torch.nn.Module):
    """ Inference wrapper for a trained Backbone """

    requires_backbone = True

    def __init__(self, args = None, backbone = None):
        super(NeuralNetAgent, self).__init__()
        self.backbone = backbone

    @staticmethod
    def add_argparse_args(parser):
        Backbone.add_argparse_args(parser)
        return parser

    @staticmethod
    def from_argparse_args(args):
        backbone = Backbone.from_argparse_args(args)
        return NeuralNetAgent(args, backbone = backbone)

    def reset(self, env):
        self.goal_state = env.goal_state
        self.obstacle_centers = env.obstacle_centers
        self.obstacle_radii = env.obstacle_radii

    def update_backbone(self, backbone):
        self.backbone.load_state_dict(backbone.state_dict())

    def get_action(self, state: np.ndarray) -> np.ndarray:
        relative_goal = state[:3] - self.goal_state[:3]
        trunc_state = state[3:]
        inputs_np = np.concatenate([trunc_state, relative_goal], axis=0)
        inputs = torch.from_numpy(inputs_np).float().view(1,-1)
        
        nn_action = self.backbone(inputs)
        nn_action = nn_action.detach().clone().numpy()[0]    
        return nn_action
