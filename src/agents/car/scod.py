from src.runners.data import CarDataset
import torch
import numpy as np

from .backbone import Backbone 
from .scp import SCPAgent

from typing import Dict

class SCODNetAgent:
    requires_backbone = True

    def __init__(self, args = None, backbone = None):
        """ Uncertainty-aware inference wrapper for a trained Backbone """
        super(SCODNetAgent, self).__init__()

        self.backbone = backbone
        self.dataset = CarDataset(args.data_filepath)
        self.oracle = SCPAgent.from_argparse_args(args)
        self.max_uncertainty = args.max_uncertainty

        self.update_backbone(backbone)

    @staticmethod
    def add_argparse_args(parser):
        Backbone.add_argparse_args(parser)
        SCPAgent.add_argparse_args(parser)
        parser.add_argument('--data-filepath', type = str, default = None)
        parser.add_argument('--max-uncertainty', type = float, default = 1)
        return parser

    @staticmethod
    def from_argparse_args(args):
        backbone = Backbone.from_argparse_args(args)
        return SCODNetAgent(args, backbone = backbone)

    def reset(self, env):
        self.goal_state = env.goal_state
        self.oracle.reset(env)

    def update_backbone(self, backbone):
        from nn_ood.posteriors import SCOD
        from nn_ood.distributions import GaussianFixedDiagVar
        self.backbone.load_state_dict(backbone.state_dict())
        self.backbone_wrapper = SCOD(
            backbone,
            GaussianFixedDiagVar()
        )
        self.backbone_wrapper.process_dataset(self.dataset)

    def get_action(self, state: np.ndarray) -> np.ndarray:
        relative_goal = state[:3] - self.goal_state[:3]
        trunc_state = state[3:]
        inputs_np = np.concatenate([trunc_state, relative_goal], axis=0)
        inputs = torch.from_numpy(inputs_np).float().view(1,-1)
        
        nn_action, uncertainty = self.backbone_wrapper(inputs)
        nn_action = nn_action.detach().clone().numpy()    
        uncertainty = uncertainty.detach().clone().numpy()    

        print(uncertainty)

        if uncertainty > self.max_uncertainty:
            return self.oracle.get_action(state)
        else:
            return nn_action