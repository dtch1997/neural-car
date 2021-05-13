import numpy as np
import torch

class NeuralNetAgent(torch.nn.Module):

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        """
        Args:
            obs_size: observation/state size of the environment
            n_actions: number of discrete actions available in the environment
            hidden_size: size of hidden layers
        """
        super(NeuralNetAgent, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, action_dim),
        )

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
            action_dim = args.action_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def reset(self, env):
        pass

    def get_action(self, state: np.ndarray) -> np.ndarray:
        state_tensor = torch.from_numpy(state.astype(np.float32))
        return self(state_tensor).detach().clone().numpy()