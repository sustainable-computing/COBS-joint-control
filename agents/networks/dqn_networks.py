import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class DeepQSequential(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        if not os.path.exists(chkpt_dir): os.makedirs(chkpt_dir)
        self.chkpt_dir = chkpt_dir
        self.name = name

        super(DeepQSequential, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dims[0], 50),
            nn.ReLU(),
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 400),
            nn.ReLU(),
            nn.Linear(400, n_actions)
        )

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)

#        self.apply(weights_init_)

        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x1 = self.model(x.float())
        return x1

    def save_checkpoint(self, num):
        checkpoint_file = os.path.join(self.chkpt_dir, self.name + f'_{num}')
        print('... saving checkpoint ...')
        T.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, num):
        checkpoint_file = os.path.join(self.chkpt_dir, self.name + f'_{num}')
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(checkpoint_file))


class DeepQ(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        if not os.path.exists(chkpt_dir): os.makedirs(chkpt_dir)
        self.chkpt_dir = chkpt_dir
        self.name = name

        hidden_dim = 256

        super(DeepQ, self).__init__()

        # Base Network
        self.linear1 = nn.Linear(input_dims[0], hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, hidden_dim)

        # Setpoint Branch
        self.linear_stpt = nn.Linear(hidden_dim, n_actions)

#        self.apply(weights_init_)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)

        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x1 = self.linear1(x.float())
        x1 = self.linear4(x1)
        stpt = self.linear_stpt(x1)
        return stpt

    def save_checkpoint(self, num):
        checkpoint_file = os.path.join(self.chkpt_dir, self.name + f'_{num}')
        print('... saving checkpoint ...')
        T.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, num):
        checkpoint_file = os.path.join(self.chkpt_dir, self.name + f'_{num}')
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(checkpoint_file))


class DeepQLeaky(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        if not os.path.exists(chkpt_dir): os.makedirs(chkpt_dir)
        self.chkpt_dir = chkpt_dir
        self.name = name

        hidden_dim = 256

        super(DeepQLeaky, self).__init__()

        # Base Network
        self.linear1 = nn.Linear(input_dims[0], hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, hidden_dim)

        # Setpoint Branch
        self.linear_stpt = nn.Linear(hidden_dim, n_actions)

#        self.apply(weights_init_)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)

        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x1 = F.leaky_relu(self.linear1(x.float()))
        x1 = F.leaky_relu(self.linear4(x1))
        stpt = self.linear_stpt(x1)
        return stpt

    def save_checkpoint(self, num):
        checkpoint_file = os.path.join(self.chkpt_dir, self.name + f'_{num}')
        print('... saving checkpoint ...')
        T.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, num):
        checkpoint_file = os.path.join(self.chkpt_dir, self.name + f'_{num}')
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(checkpoint_file))

