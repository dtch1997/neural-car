import numpy as np
import h5py

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split

class CarDataset(Dataset):
    def __init__(self, data_filepath, transform=None, target_transform=None):
        self.dataset = h5py.File(data_filepath, 'r')['simulation_0']

    def __len__(self):
        return self.dataset.attrs['num_total_steps']

    def __getitem__(self, idx):
        goal = 0
        for current_goal, end_marker in enumerate(self.dataset.attrs['end_markers']):
            if end_marker >= idx:
                goal = current_goal
                break

        start_marker = self.dataset.attrs['end_markers'][goal-1] if goal > 0 else 0
        traj_time = idx-start_marker-1 if start_marker > 0 else idx

        sub_data = self.dataset[f'goal_{goal}']
        current_state = torch.from_numpy(sub_data['state_trajectory'][traj_time+1,:].astype(np.float32))
        action = torch.from_numpy(sub_data['input_trajectory'][traj_time,:].astype(np.float32))
        goal_state = torch.from_numpy(sub_data.attrs['goal_state'].astype(np.float32))

        trunc_state = current_state[3:]
        relative_goal = current_state[:3] - goal_state[:3]
        inp = torch.cat([relative_goal, trunc_state])
        oup = action

        return inp, oup

class CarDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
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
