import os

import torch as T
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class DeepQSequential_Blinds(nn.Module):
    def __init__(self, lr, n_stpt_actions, n_blind_actions, hidden_dim, name, input_dims, chkpt_dir):
        if not os.path.exists(chkpt_dir): os.makedirs(chkpt_dir)
        self.chkpt_dir = chkpt_dir
        self.name = name

        super(DeepQSequential_Blinds, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dims[0], 50),
            nn.ReLU(),
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 400),
            nn.ReLU(),
            # nn.Linear(400, n_actions)
        )

        self.linear_stpt = nn.Linear(400, n_stpt_actions)
        self.linear_blind = nn.Linear(400, n_blind_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)

#        self.apply(weights_init_)

        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x1 = self.model(x.float())
        stpt = self.linear_stpt(x1)
        blnd = self.linear_blind(x1)
        return stpt, blnd

    def save_checkpoint(self, num):
        checkpoint_file = os.path.join(self.chkpt_dir, self.name + f'_{num}')
        print('... saving checkpoint ...')
        T.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, num):
        checkpoint_file = os.path.join(self.chkpt_dir, self.name + f'_{num}')
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(checkpoint_file))


class DeepQ_Blinds(nn.Module):
    def __init__(self, lr, n_stpt_actions, n_blind_actions, hidden_dim, name, input_dims, chkpt_dir):
        if not os.path.exists(chkpt_dir): os.makedirs(chkpt_dir)
        self.chkpt_dir = chkpt_dir
        self.name = name

        super(DeepQ_Blinds, self).__init__()

        # Base Network
        self.linear1 = nn.Linear(input_dims[0], hidden_dim)
        # self.linear2 = nn.Linear(hidden_dim//2, hidden_dim)
        # self.linear3 = nn.Linear(hidden_dim, hidden_dim*2)
        self.linear4 = nn.Linear(hidden_dim, hidden_dim)

        # Setpoint Branch
        self.linear_stpt = nn.Linear(hidden_dim, n_stpt_actions)

        # Blind Branch
        self.linear_blind = nn.Linear(hidden_dim, n_blind_actions)

  #      self.apply(weights_init_)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)

        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        # Base Network
        x1 = self.linear1(x.float())
        x1 = self.linear4(x1)

        # Branches
        stpt = self.linear_stpt(x1)
        blind = self.linear_blind(x1)

        return stpt, blind

    def save_checkpoint(self, num):
        checkpoint_file = os.path.join(self.chkpt_dir, self.name + f'_{num}')
        print('... saving checkpoint ...')
        T.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, num):
        checkpoint_file = os.path.join(self.chkpt_dir, self.name + f'_{num}')
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(checkpoint_file))


class DeepQLeaky_Blinds(nn.Module):
    def __init__(self, lr, n_stpt_actions, n_blind_actions, hidden_dim, name, input_dims, chkpt_dir):
        if not os.path.exists(chkpt_dir): os.makedirs(chkpt_dir)
        self.chkpt_dir = chkpt_dir
        self.name = name

        super(DeepQLeaky_Blinds, self).__init__()

        # Base Network
        self.linear1 = nn.Linear(input_dims[0], hidden_dim)
        # self.linear2 = nn.Linear(hidden_dim//2, hidden_dim)
        # self.linear3 = nn.Linear(hidden_dim, hidden_dim*2)
        self.linear4 = nn.Linear(hidden_dim, hidden_dim)

        # Setpoint Branch
        self.linear_stpt = nn.Linear(hidden_dim, n_stpt_actions)

        # Blind Branch
        self.linear_blind = nn.Linear(hidden_dim, n_blind_actions)

 #       self.apply(weights_init_)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)

        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        # Base Network
        x1 = F.leaky_relu(self.linear1(x.float()))
        x1 = F.leaky_relu(self.linear4(x1))

        # Branches
        stpt = self.linear_stpt(x1)
        blind = self.linear_blind(x1)

        return stpt, blind

    def save_checkpoint(self, num):
        checkpoint_file = os.path.join(self.chkpt_dir, self.name + f'_{num}')
        print('... saving checkpoint ...')
        T.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, num):
        checkpoint_file = os.path.join(self.chkpt_dir, self.name + f'_{num}')
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(checkpoint_file))
