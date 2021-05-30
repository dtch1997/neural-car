from src.runners.data import CarDataset
import torch
import numpy as np

from .backbone import Backbone 
from .scp import SCPAgent

from nn_ood.posteriors import SCOD
from nn_ood.distributions import GaussianFixedDiagVar

class SCODNetAgent:
    def __init__(self, args):
        """
        Args:
            obs_size: observation/state size of the environment
            n_actions: number of discrete actions available in the environment
            hidden_size: size of hidden layers
        """
        super(SCODNetAgent, self).__init__()
        
        self.backbone = SCOD(
            Backbone.from_argparse_args(args),
            GaussianFixedDiagVar()
        )
        dataset = CarDataset(args.data_filepath)
        self.backbone.process_dataset(dataset)
        self.oracle = SCPAgent.from_argparse_args(args)
        self.uncertainty_threshold = args.uncertainty_threshold

    @staticmethod
    def add_argparse_args(parser):
        Backbone.add_argparse_args(parser)
        SCPAgent.add_argparse_args(parser)
        parser.add_argument('--data-filepath', type = str, default = None)
        parser.add_argument('--uncertainty-threshold', type = float, default = 0.5)
        return parser

    @staticmethod
    def from_argparse_args(args):
        return SCODNetAgent(args)

    def forward(args):
        raise NotImplementedError("SCODNetAgent.forward() should not be called." 
                                "Use NeuralNetAgent to train the model"
                                "or SCODNetAgent.get_action() to perform inference")

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
        
        nn_action, uncertainty = self.backbone(inputs)
        nn_action = nn_action.detach().clone().numpy()    
        uncertainty = uncertainty.detach().clone().numpy()    

        if uncertainty > self.max_uncertainty:
            return self.oracle.get_action(state)
        else:
            return nn_action