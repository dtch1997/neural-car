import numpy as np
import torch
import pytorch_lightning as pl
from .data import CarDataModule

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
        inputs, targets = batch
        
        action = targets
        action_pred = self.agent(inputs)
        # Compute MSE loss, averaging over samples
        loss = torch.nn.functional.mse_loss(action_pred, action, reduction = 'mean')
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
        self.trainer = pl.Trainer(auto_lr_find=True)

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

    

    