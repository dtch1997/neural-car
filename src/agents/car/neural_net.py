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
        self.state_trunc_adapter = torch.nn.Linear(4,hidden_dim)
        self.relative_goal_adapter = torch.nn.Linear(3, hidden_dim)
        self.obstacle_center_adapter = torch.nn.Linear(2, hidden_dim)
        self.obstacle_radius_adapter = torch.nn.Linear(1, hidden_dim)

        self.hidden_1 = torch.nn.Linear(hidden_dim*2, hidden_dim)
        self.hidden_2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.hidden_3 = torch.nn.Linear(hidden_dim, action_dim)

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
        state_embedding_trunc = self.state_trunc_adapter(sample['trunc_state']) #modified to no longer include x,y,theta
        goal_embedding = self.relative_goal_adapter(sample['relative_goal'])
        embeddings = torch.cat((state_embedding_trunc,goal_embedding),dim=-1)
        embeddings = torch.tanh(embeddings)
        hidden_out_1 = torch.tanh(self.hidden_1(embeddings))
        hidden_out_2 = torch.tanh(self.hidden_2(hidden_out_1))
        action = self.hidden_3(hidden_out_2)

        return action

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
        return self(inputs).detach().clone().numpy()
