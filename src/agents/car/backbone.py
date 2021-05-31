import torch
import torch.nn.functional as F

from typing import Dict

class Backbone(torch.nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        """
        Args:
            obs_size: observation/state size of the environment
            n_actions: number of discrete actions available in the environment
            hidden_size: size of hidden layers
        """
        super(Backbone, self).__init__()
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
        return Backbone(
            state_dim = args.state_dim,
            hidden_dim = args.hidden_dim,
            action_dim = args.action_dim,
        )

    def forward(self, sample: Dict[str, torch.Tensor]) -> torch.Tensor:
        state_embedding_trunc = self.state_trunc_adapter(sample['trunc_state']) #modified to no longer include x,y,theta
        goal_embedding = self.relative_goal_adapter(sample['relative_goal'])
        embeddings = torch.cat((state_embedding_trunc, goal_embedding),dim=-1)
        embeddings = torch.tanh(embeddings)
        hidden_out_1 = torch.tanh(self.hidden_1(embeddings))
        hidden_out_2 = torch.tanh(self.hidden_2(hidden_out_1))
        action = self.hidden_3(hidden_out_2)
        return action 

    def reset(self, env):
        self.goal_state = env.goal_state
        self.obstacle_centers = env.obstacle_centers
        self.obstacle_radii = env.obstacle_radii

    """
    def save_to_checkpoint(self, filepath: str):
        torch.save(self.state_dict(), filepath)
    
    @staticmethod 
    def load_from_checkpoint(filepath: str, *args, **kwargs):
        backbone = Backbone(*args, **kwargs)
        backbone.load_state_dict(filepath)
        return backbone
    """