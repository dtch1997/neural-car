import torch
import numpy as np
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader, random_split

class CarDataset(Dataset):
    def __init__(self, data_filepath, transform=None, target_transform=None):
        data = np.load(open(data_filepath), delimiter=",")
        self.state_data = data[:,:-3]
        self.action_data = data[:,-3:]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.state_data.shape[0]

    def __getitem__(self, idx):
        state = torch.from_numpy(self.state_data[idx,:])
        action = torch.from_numpy(self.action_data[idx,:])
        if self.transform:
            state = self.transform(state)
        if self.target_transform:
            action = self.target_transform(action)
        sample = {"state": state, "action": action}
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
        parser.add_argument('--batch-size', type = int, default = 32)
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
        self.train_dataset, self.val_dataset = random_split(self.dataset, train_len, val_len, 
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
    def __init__(self, args, agent: 'NeuralNetAgent'):
        super().__init__()
        self.agent = agent

    @staticmethod 
    def add_argparse_args(parser):
        return parser

    @staticmethod
    def from_argparse_args(args, agent):
        return MSERegression(args, agent)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)
    
    def shared_step(self, batch, batch_idx):
        """ The common parts of train_step and validation_step """
        state = batch['state']
        action = batch['action']
        action_pred = self.agent(state)
        # Compute MSE loss, averaging over samples
        loss = torch.nn.functional.mse_loss(action_pred, action, reduction = 'mean')
        return {'loss': loss}

    def training_step(self, batch, batch_idx):
        output = self.shared_step(batch, batch_idx)
        self.log('train_loss', output['loss'])
        return output
        
    def validation_step(self, batch, batch_idx):
        output = self.shared_step(batch, batch_idx)
        self.log('val_loss', output['loss'])
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
        parser.add_argument('--learning-rate', default = None)
        parser.add_argument('--epochs', default = None)
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

    

    