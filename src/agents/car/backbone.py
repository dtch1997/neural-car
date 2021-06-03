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
        self.emb = torch.nn.Linear(state_dim, hidden_dim)
        self.h1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.h2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.out = torch.nn.Linear(hidden_dim, action_dim)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.emb(x))
        x = torch.tanh(self.h1(x))
        x = torch.tanh(self.h2(x))
        return self.out(x)