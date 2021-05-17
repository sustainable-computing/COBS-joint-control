import os
import torch.nn as nn
from torch import save, load, sqrt, zeros, device, cuda


def init_layer(m):
    weight = m.weight.data
    weight.normal_(0, 1)
    weight *= 1.0 / sqrt(weight.pow(2).sum(1, keepdim=True))
    nn.init.constant_(m.bias.data, 0)
    return m


class ActorNetwork(nn.Module):
    def __init__(self, n_actions, name, input_dims, chkpt_dir, network_type):
        if not os.path.exists(chkpt_dir):
            os.makedirs(chkpt_dir)
        self.chkpt_dir = chkpt_dir
        self.name = name
        self.device = device("cuda:0" if cuda.is_available() else "cpu")

        super(ActorNetwork, self).__init__()

        if isinstance(n_actions, int):
            n_actions = [range(n_actions)]

        relu = nn.ReLU if network_type != "leaky" else nn.LeakyReLU
        activation = relu  # if discrete else nn.Tanh

        self.model = nn.Sequential(
            nn.Linear(input_dims, 100),
            activation(),
            nn.Linear(100, 100),
            activation(),
        ).to(self.device)
        if network_type == "no_relu":
            self.model = nn.Sequential(
                nn.Linear(input_dims, 100),
                nn.Linear(100, 100),
            ).to(self.device)

        self.output = list()
        for action_space in n_actions:
            i = len(action_space)
            self.output.append(nn.Sequential(
                nn.Linear(100, i),
                nn.Softmax(dim=-1)).to(self.device))

    def forward(self, x):
        x1 = self.model(x.float())
        if not self.output:
            return [x1]
        result = list()
        for network in self.output:
            result.append(network(x1))
        return result

    def save_checkpoint(self, num):
        checkpoint_file = os.path.join(self.chkpt_dir, self.name + f'_{num}')
        print('... saving checkpoint ...')
        save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, num):
        checkpoint_file = os.path.join(self.chkpt_dir, self.name + f'_{num}')
        print('... loading checkpoint ...')
        self.load_state_dict(load(checkpoint_file))


class CriticNetwork(nn.Module):
    def __init__(self, name, input_dims, chkpt_dir):
        if not os.path.exists(chkpt_dir):
            os.makedirs(chkpt_dir)
        self.chkpt_dir = chkpt_dir
        self.name = name

        super(CriticNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dims, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
        )

    def forward(self, x):
        x1 = self.model(x.float())
        return x1

    def save_checkpoint(self, num):
        checkpoint_file = os.path.join(self.chkpt_dir, self.name + f'_{num}')
        print('... saving checkpoint ...')
        save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, num):
        checkpoint_file = os.path.join(self.chkpt_dir, self.name + f'_{num}')
        print('... loading checkpoint ...')
        self.load_state_dict(load(checkpoint_file))
