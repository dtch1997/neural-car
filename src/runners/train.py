import numpy as np
import h5py

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split

class CarDataset(Dataset):
    def __init__(self, data_filepath, transform=None, target_transform=None):
        self.dataset = h5py.File(data_filepath, 'r')['simulation_0']

    def __len__(self):
        return self.dataset.attrs['num_steps']

    def __getitem__(self, idx):
        current_state = torch.from_numpy(self.dataset['state_trajectory'][idx,:].astype(np.float32))
        action = torch.from_numpy(self.dataset['input_trajectory'][idx,:].astype(np.float32))
        goal_state = torch.from_numpy(self.dataset.attrs['goal_state'].astype(np.float32))
        obstacle_centers = torch.from_numpy(self.dataset.attrs['obstacle_centers'].astype(np.float32))
        obstacle_radii = torch.from_numpy(self.dataset.attrs['obstacle_radii'].astype(np.float32))

        sample = {
            "state": current_state, 
            "action": action,
            "relative_goal": current_state[:3] - goal_state[:3],
            "obstacle_centers": current_state[:2] - obstacle_centers,
            "obstacle_radii": obstacle_radii, 
        }
        return sample

class CarDataModule(pl.LightningDataModule):
    def __init__(self, args):
        self.data_filepath = args.data_filepath
        self.train_fraction = args.train_fraction
        self.data_seed = args.data_seed
        self.batch_size = args.batch_size

    @staticmethod 
    def add_argparse_args(parser):
        parser.add_argument('--data-filepath', default = None)
        parser.add_argument('--batch-size', type = int, default = 8)
        parser.add_argument('--train-fraction', type = float,  default = 0.8)
        parser.add_argument('--data-seed', type = int, default = 0)
        return parser 

    @staticmethod 
    def from_argparse_args(args):
        return CarDataModule(args)

    def prepare_data(self):
        self.dataset = CarDataset(self.data_filepath)

    def setup(self):
        train_len = int(self.train_fraction * len(self.dataset))
        val_len = len(self.dataset) - train_len
        self.train_dataset, self.val_dataset = random_split(self.dataset, 
            lengths = (train_len, val_len), 
            generator = torch.Generator().manual_seed(self.data_seed))

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
            batch_size = self.batch_size, 
            shuffle = True
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
            batch_size = self.batch_size, 
            shuffle = False
        )

class MSERegression(pl.LightningModule):
    """ A pl.Lightning module for training a module to minimize MSE loss between source and target """
    def __init__(self, args = None, agent: 'NeuralNetAgent' = None):
        super().__init__()
        self.agent = agent

    @staticmethod 
    def add_argparse_args(parser):
        return parser

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.agent.forward(x)

    def reset(self, env):
        self.agent.reset(env)

    def get_action(self, state: np.ndarray) -> np.ndarray:
        return self.agent.get_action(state)

    @staticmethod
    def from_argparse_args(args, agent):
        return MSERegression(args, agent)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)
    
    def shared_step(self, batch, batch_idx):
        """ The common parts of train_step and validation_step """
        inputs = {
            'state': batch['state'],
            'relative_goal': batch['relative_goal'],
            'obstacle_centers': batch['obstacle_centers'],
            'obstacle_radii': batch['obstacle_radii']
        }
        action = batch['action']
        action_pred = self.agent(inputs)
        # Compute MSE loss, averaging over samples
        loss = torch.nn.functional.l1_loss(action_pred, action, reduction = 'mean')
        # Compute relative deviation 
        avg_relative_deviation = (torch.abs(action_pred - action) / action).mean()
        return {'loss': loss, 'relative_deviation': avg_relative_deviation}

    def training_step(self, batch, batch_idx):
        output = self.shared_step(batch, batch_idx)
        self.log('train_loss', output['loss'])
        self.log('train_deviation', output['relative_deviation'])
        return output
        
    def validation_step(self, batch, batch_idx):
        output = self.shared_step(batch, batch_idx)
        self.log('val_loss', output['loss'])
        self.log('val_deviation', output['relative_deviation'])
        return output

class TrainingRunner:
    def __init__(self, args, env, agent):
        self.env = env
        self.agent = agent
        self.learning_rate = args.learning_rate
        self.epochs = args.epochs

        self.data_module = CarDataModule.from_argparse_args(args)
        self.model = MSERegression.from_argparse_args(args, agent)
        self.trainer = pl.Trainer()

    @staticmethod 
    def add_argparse_args(parser):
        parser.add_argument('--learning-rate', default = 0.01)
        parser.add_argument('--epochs', default = 1000)
        parser = MSERegression.add_argparse_args(parser)
        parser = CarDataModule.add_argparse_args(parser)
        return parser 

    @staticmethod 
    def from_argparse_args(args, env, agent):
        return TrainingRunner(args, env, agent)

    def run(self):
        self.data_module.prepare_data()
        self.data_module.setup()
        self.trainer.fit(self.model, self.data_module)

    

    