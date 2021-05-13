import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np

class CarDataset(Dataset):
    def __init__(self, data_filepath, transform=None, target_transform=None):
        data = np.loadtxt(open(data_filepath), delimiter=",")
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

class TrainingRunner:
    def __init__(self, args, env, agent):
        self.env = env
        self.agent = agent
        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        self.epochs = args.epochs

    @staticmethod 
    def add_argparse_args(parser):
        parser.add_argument('--learning-rate', default = None)
        parser.add_argument('--batch-size', default = None)
        parser.add_argument('--epochs', default = None)
        parser.add_argument('--data-filepath', default = None)
        return parser 

    @staticmethod 
    def from_argparse_args(args, env, agent):
        return TrainingRunner(args, env, agent)
    
    def train_loop(self, dataloader, loss_fn, optimizer):
        size = len(dataloader.dataset)
        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            pred = self.agent(X)
            loss = loss_fn(pred, y)
    
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def run(self):
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(self.agent.parameters(), lr=self.learning_rate)
        training_data = CarDataset(data_filepath='scp_data')
        train_dataloader = DataLoader(training_data, batch_size=self.batch_size, shuffle=True)

        for t in range(self.epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            self.train_loop(train_dataloader, loss_fn, optimizer)

    

    