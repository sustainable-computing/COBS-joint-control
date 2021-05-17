# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import glob
# import os
# from tqdm import tqdm
#
# # plt.figure(figsize=(10, 8))
# for i, chunk in tqdm(enumerate(pd.read_csv(f"rl_results/THERM_SP_MultiTrue_customOccFalse_PPO_leaky_cooling_
# blindsTrueMultiTrue_dlightingFalse_multiAgentTrue_0.1_0.1_0.1/run.csv", chunksize=3071)), total=600):
#     print(f"HVAC Power SUM: {chunk['HVAC Power'].sum()}\n"
#           f"Heating Coil Power SUM: {chunk['Heating Coil Power'].sum()}\n"
#           f"Cooling Coil Power SUM: {chunk['Cooling Coil Power'].sum()}\n"
#           f"Fan Power SUM: {chunk['Fan Power'].sum()}\n"
#           f"Heating + Cooling + Fan: {chunk['Heating Coil Power'].sum() + chunk['Cooling Coil Power'].sum()
#           + chunk['Fan Power'].sum()}\n")
#
# # plt.plot(result)
# # plt.show()

# import numpy as np
#
# Q = 25
# a_t = 0.02
# b_t = 0.05
#
# a = -100
# b = -100
#
# total_serv = 0
# serv_count = 0
#
# for current_time, num in enumerate(np.random.poisson(Q, 500)):
#     for t in range(num):
#         ct = t * 1 / num + current_time
#
#         a_serv = a_t
#         b_serv = np.random.exponential(b_t, 1)
#
#         if a < ct and b < ct:
#             if np.random.random() < 0.5:
#                 a = ct + a_serv
#                 total_serv += a_serv
#                 serv_count += 1
#             else:
#                 b = ct + b_serv
#                 total_serv += b_serv
#                 serv_count += 1
#         elif a < ct:
#             a = ct + a_serv
#             total_serv += a_serv
#             serv_count += 1
#         elif b < ct:
#             b = ct + b_serv
#             total_serv += b_serv
#             serv_count += 1
#         else:
#             print(1)
# print(total_serv / serv_count)

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import utils as utils
import os
import random
import numpy as np
from collections import deque
from tqdm import tqdm

import matplotlib.pyplot as plt


# ### Q Network of CartPole Env

class QNetwork(nn.Module):

    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.tanh(self.fc1(x.to(device)))
        q_values = self.fc2(x)

        return q_values


def get_action(q_values, action_size, epsilon):
    if np.random.rand() <= epsilon:
        return random.randrange(action_size)
    else:
        _, action = torch.max(q_values, 1)
        return action.cpu().numpy()[0]


def update_model_parameters(q_net, target_q_net, optimizer, mini_batch):

    mini_batch = np.array(mini_batch)
    states = np.vstack(mini_batch[:, 0])
    actions = list(mini_batch[:, 1])
    rewards = list(mini_batch[:, 2])
    next_states = np.vstack(mini_batch[:, 3])
    masks = list(mini_batch[:, 4])

    states = torch.Tensor(states).to(device)
    actions = torch.LongTensor(actions).to(device)
    rewards = torch.Tensor(rewards).to(device)
    masks = torch.Tensor(masks).to(device)

    criterion = torch.nn.MSELoss()

    # get Q-value
    q_values = q_net(states)
    q_value = q_values.gather(1, actions.unsqueeze(1)).view(-1)

    # get target
    target_next_q_values = target_q_net(torch.Tensor(next_states))
    target = rewards + masks * gamma * target_next_q_values.max(1)[0]

    loss = criterion(q_value, target.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def update_target_network(q_net, target_q_net):
    target_q_net.load_state_dict(q_net.state_dict())


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make(env_name)
    env.seed(500)
    torch.manual_seed(500)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    print('state size:', state_size)
    print('action size:', action_size)

    q_net = QNetwork(state_size, action_size).to(device)
    target_q_net = QNetwork(state_size, action_size).to(device)
    update_target_network(q_net, target_q_net)

    optimizer = optim.Adam(q_net.parameters(), lr=0.001)

    replay_buffer = deque(maxlen=10000)

    running_score = 0
    steps = 0
    epsilon = 0.05

    sum_of_rewards = np.zeros(max_iter_num)

    for episode in tqdm(range(max_iter_num)):
        done = False
        score = 0

        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            if render:
                env.render()

            steps += 1

            q_values = q_net(torch.Tensor(state).to(device))
            action = get_action(q_values, action_size, epsilon)

            next_state, reward, done, _ = env.step(action)

            next_state = np.reshape(next_state, [1, state_size])

            if (not done) or (score == 499):
                reward = reward
            else:
                reward = -1

            if done:
                mask = 0
            else:
                mask = 1

            replay_buffer.append((state, action, reward, next_state, mask))

            state = next_state
            score += reward

            if steps > initial_exploration:
                epsilon -= epsilon_decay
                epsilon = max(epsilon, 0.1)

                mini_batch = random.sample(replay_buffer, batch_size)

                q_net.train()
                target_q_net.train()

                update_model_parameters(q_net, target_q_net, optimizer,
                                        mini_batch)  # same as update parameter step of Saidur's SAC

                if steps % update_target == 0:
                    update_target_network(q_net, target_q_net)

            if (score == 500):
                score = score
            else:
                score += 1

            sum_of_rewards[episode] += reward

    return sum_of_rewards


# In[10]:


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    initial_exploration = 4
    epsilon = 0.05
    batch_size = 1
    epsilon_decay = 0.001
    update_target = 10
    render = True
    max_iter_num = 50
    env_name = "CartPole-v0"
    gamma = 0.9
    hidden_size = 256
    rez = main()

