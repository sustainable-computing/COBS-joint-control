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


class BranchingDuelingDeepQOcto(nn.Module):
    def __init__(self, lr, n_stpt_actions, n_therm_actions, n_blind_actions,
                 nd_stpt_actions, nd_therm_actions, nd_blind_actions,
                 name, input_dims, chkpt_dir):
        if not os.path.exists(chkpt_dir): os.makedirs(chkpt_dir)
        self.chkpt_dir = chkpt_dir
        self.name = name

        super(BranchingDuelingDeepQOcto, self).__init__()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        # self.to(self.device)

        self.n_stpt_actions = n_stpt_actions
        self.n_therm_actions = n_therm_actions
        self.n_blind_actions = n_blind_actions

        # Base Network
        self.linear1 = nn.Linear(input_dims[0], 512).to(self.device)
        self.linear2 = nn.Linear(512, 256).to(self.device)
        self.linear3 = nn.Linear(256, 128).to(self.device)

        # Value estimate
        self.V = nn.Linear(128, 1).to(self.device)

        # Action estimates
        for i in range(n_stpt_actions):
            setattr(self, f"A_stpt_{i}", nn.Linear(128, nd_stpt_actions).to(self.device))
        for i in range(n_therm_actions):
            setattr(self, f"A_therm_{i}", nn.Linear(128, nd_therm_actions).to(self.device))
        for i in range(n_blind_actions):
            setattr(self, f"A_blind_{i}", nn.Linear(128, nd_blind_actions).to(self.device))
        self.apply(weights_init_)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

    def forward(self, x):
        x1 = F.relu(self.linear1(x.float()))
        x1 = F.relu(self.linear2(x1))
        x1 = F.relu(self.linear3(x1))

        V = self.V(x1)

        A_stpts = []
        A_therms = []
        A_blinds = []
        for i in range(self.n_stpt_actions):
            A = getattr(self, f"A_stpt_{i}")
            A_stpts.append(A(x1))
        for i in range(self.n_therm_actions):
            A = getattr(self, f"A_therm_{i}")
            A_therms.append(A(x1))
        for i in range(self.n_blind_actions):
            A = getattr(self, f"A_blind_{i}")
            A_blinds.append(A(x1))

        return V, A_stpts, A_therms, A_blinds

    def save_checkpoint(self, num):
        checkpoint_file = os.path.join(self.chkpt_dir, self.name + f'_{num}')
        print('... saving checkpoint ...')
        T.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, num):
        checkpoint_file = os.path.join(self.chkpt_dir, self.name + f'_{num}')
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(checkpoint_file))

