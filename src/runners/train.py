import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from src.agents.car.backbone import Backbone
from .data import CarDataModule

class TrainingWrapper(pl.LightningModule):
    """ A pl.Lightning module for training a module to minimize regression loss between source and target """
    def __init__(self, loss_fn: str = 'l1', backbone: Backbone = None):
        super().__init__()
        loss_fn_factory = ({
            'l1': torch.nn.L1Loss,
            'l2': torch.nn.MSELoss
        })[loss_fn]

        self.loss_fn = loss_fn_factory(reduction = 'mean')
        self.backbone = backbone

    @staticmethod 
    def add_argparse_args(parser):
        parser.add_argument('--loss-fn', type = str, default = 'l1', choices = ('l1', 'l2'))
        return parser

    @staticmethod
    def from_argparse_args(args, backbone):
        return TrainingWrapper(args.loss_fn, backbone)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)
    
    def shared_step(self, batch, batch_idx):
        """ The common parts of train_step and validation_step """
        inputs, targets = batch
        
        action = targets
        action_pred = self.backbone(inputs)
        # Compute MSE loss, averaging over samples
        loss = self.loss_fn(action_pred, action)
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
        self.learning_rate = args.learning_rate
        self.epochs = args.epochs

        self.data_module = CarDataModule.from_argparse_args(args)
        self.model = TrainingWrapper.from_argparse_args(args, agent.backbone)
        self.trainer = pl.Trainer(auto_lr_find=True)

    @staticmethod 
    def add_argparse_args(parser):
        parser.add_argument('--learning-rate', default = 0.01)
        parser.add_argument('--epochs', default = 1000)
        parser = TrainingWrapper.add_argparse_args(parser)
        parser = CarDataModule.add_argparse_args(parser)
        return parser 

    @staticmethod 
    def from_argparse_args(args, env, agent):
        return TrainingRunner(args, env, agent)

    def run(self):
        self.data_module.prepare_data()
        self.data_module.setup()
        self.trainer.fit(self.model, self.data_module)

    

    