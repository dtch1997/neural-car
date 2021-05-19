import numpy as np
import torch
import torch.nn.functional as F

from typing import Dict

class NeuralNetAgent(torch.nn.Module):

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        """
        Args:
            obs_size: observation/state size of the environment
            n_actions: number of discrete actions available in the environment
            hidden_size: size of hidden layers
        """
        super(NeuralNetAgent, self).__init__()
        self.state_adapter = torch.nn.Linear(state_dim, hidden_dim)
        self.relative_goal_adapter = torch.nn.Linear(3, hidden_dim)
        self.obstacle_center_adapter = torch.nn.Linear(2, hidden_dim)
        self.obstacle_radius_adapter = torch.nn.Linear(1, hidden_dim)

        self.hidden_1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.hidden_2 = torch.nn.Linear(hidden_dim, action_dim)

    @staticmethod 
    def add_argparse_args(parser):
        parser.add_argument('--state-dim', type=int, default=7)
        parser.add_argument('--hidden-dim', type=int, default=32)
        parser.add_argument('--action-dim', type=int, default=2)
        return parser 

    @staticmethod 
    def from_argparse_args(args):
        return NeuralNetAgent(
            state_dim = args.state_dim,
            hidden_dim = args.hidden_dim, 
            action_dim = args.action_dim,
        )

    def forward(self, sample: Dict[str, torch.Tensor]) -> torch.Tensor:
        state_embedding = self.state_adapter(sample['state'])
        goal_embedding = self.relative_goal_adapter(sample['relative_goal'])
        # Mean across the num_obstacles dimension
        obstacle_center_embedding = self.obstacle_center_adapter(sample['obstacle_centers']).mean(axis=-2)
        obstacle_radii_embedding = self.obstacle_radius_adapter(sample['obstacle_radii']).mean(axis=-2)
        embeddings = state_embedding + goal_embedding + obstacle_center_embedding + obstacle_radii_embedding 
        embeddings = F.relu(embeddings)
        hidden_out_1 = F.relu(self.hidden_1(embeddings))
        action = self.hidden_2(hidden_out_1)
        
        return action

    def reset(self, env):
        pass

    def get_action(self, state: np.ndarray) -> np.ndarray:
        state_tensor = torch.from_numpy(state.astype(np.float32))
        return self(state_tensor).detach().clone().numpy()