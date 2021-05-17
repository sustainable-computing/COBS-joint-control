import torch
import torch.nn as nn
from torch.distributions import Categorical
import pickle
import os
from utils.for_agents import augment_ma
from numpy import linspace


class ReplayMemory:
    def __init__(self, chkpt_dir='.'):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.chkpt_dir = chkpt_dir
        self.actions_blind = []
        self.logprobs_blind = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.actions_blind[:]
        del self.logprobs_blind[:]

    def save(self, num):
        checkpoint_file = os.path.join(self.chkpt_dir, f'memory_{num}')
        dump_dict = {
            'actions': self.actions,
            'states': self.states,
            'logprobs': self.logprobs,
            'rewards': self.rewards,
            'is_terminals': self.is_terminals,
            'actions_blind': self.actions_blind,
            'logprobs_blind': self.logprobs_blind
        }
        with open(checkpoint_file, "wb") as f:
            pickle.dump(dump_dict, f)

    def load(self, num):
        checkpoint_file = os.path.join(self.chkpt_dir, f'memory_{num}')
        with open(checkpoint_file, "rb") as f:
            dump_dict = pickle.load(f)
            self.actions = dump_dict['actions']
            self.states = dump_dict['states']
            self.logprobs = dump_dict['logprobs']
            self.rewards = dump_dict['rewards']
            self.is_terminals = dump_dict['is_terminals']
            self.actions_blind = dump_dict['actions_blind']
            self.logprobs_blind = dump_dict['logprobs_blind']


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dims, Networks, device, name, discrete, action_space, chkpt_dir='.'):
        super(ActorCritic, self).__init__()
        self.device = device
        self.discrete = discrete

        # actor
        self.action_network = Networks[0](action_dims, f"ppo_actor_{name}", state_dim, discrete, chkpt_dir, Networks[2])

        # critic
        self.value_network = Networks[1](f"ppo_critic_{name}", state_dim, chkpt_dir)

        self.action_mean = None
        self.action_log_std = None
        self.action_scale = None
        self.action_bias = None
        self.action_space = action_space
        if not discrete:
            self.action_mean = nn.Linear(50, 1)
            self.action_log_std = nn.Linear(50, 1)
            self.action_scale = torch.FloatTensor([(action_space[1] - action_space[0]) / 2.])
            self.action_bias = torch.FloatTensor([(action_space[1] - action_space[0]) / 2.])

    def act(self, state, memory):
        state = state.to(self.device)
        action_probs, blind_action_probs = self.action_network(state)
        if self.discrete:
            dist = Categorical(action_probs)
            action = dist.sample()
            action_scaled = action
        else:
            mean = self.action_mean(action_probs)
            log_std = self.action_log_std(action_probs)
            dist = torch.distributions.Normal(mean, log_std.exp())
            action = dist.rsample()
            action_scaled = torch.tanh(action) * self.action_scale + self.action_bias + self.action_space[0]

        dist_blind = Categorical(blind_action_probs)
        action_blind = dist_blind.sample()
        action_blind_scaled = action_blind

        memory.states.append(state)
        memory.actions.append(action)
        memory.actions_blind.append(action_blind)
        memory.logprobs.append(dist.log_prob(action))
        memory.logprobs_blind.append(dist_blind.log_prob(action_blind))

        return action_scaled.item(), action_blind_scaled.item()

    def evaluate(self, state, action, action_blind):
        action_probs, blind_action_probs = self.action_network(state)
        if self.discrete:
            dist = Categorical(action_probs)
        else:
            mean = self.action_mean(action_probs)
            log_std = self.action_log_std(action_probs)
            dist = torch.distributions.Normal(mean, log_std.exp())

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        dist_blind = Categorical(blind_action_probs)
        action_logprobs_blind = dist_blind.log_prob(action_blind)
        dist_blind_entropy = dist_blind.entropy()

        state_value = self.value_network(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy,\
               action_logprobs_blind, dist_blind_entropy

    def save_checkpoint(self, num):
        self.action_network.save_checkpoint(num)
        self.value_network.save_checkpoint(num)


class PPOAgentBlinds:
    def __init__(self, agent_info, Networks, chkpt_dir='.'):
        # print(agent_info)
        self.type = "PPOAgent"
        self.lr = agent_info['lr']
        self.gamma = agent_info['gamma']
        self.eps_clip = agent_info['clip_param']
        self.update_timestep = agent_info['step']
        self.k_epochs = agent_info['update_episode']
        state_dim = agent_info['n_state']
        action_dim = agent_info['num_actions']
        blind_action_dim = agent_info['num_blind_actions']
        self.last_action = None
        self.last_state = None
        self.discrete = agent_info['discrete']
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.min_action = agent_info['min_action']
        self.max_action = agent_info['max_action']
        self.action_space = linspace(agent_info['min_action'], agent_info['max_action'], action_dim).round()
        self.blind_action_space = linspace(0, 180, blind_action_dim).round()

        self.memory = ReplayMemory(chkpt_dir)

        self.policy = ActorCritic(state_dim, [action_dim, blind_action_dim],
                                  Networks, self.device, "policy", self.discrete,
                                  [self.min_action, self.max_action],
                                  chkpt_dir).to(self.device)
        self.optimizer = torch.optim.RMSprop(self.policy.parameters(), lr=self.lr)
        self.policy_old = ActorCritic(state_dim, [action_dim, blind_action_dim],
                                      Networks, self.device, "old_policy", self.discrete,
                                      [self.min_action, self.max_action],
                                      chkpt_dir).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.loss = nn.MSELoss()

    def update(self):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.memory.rewards), reversed(self.memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.stack(self.memory.states).to(self.device).detach()
        old_actions = torch.stack(self.memory.actions).to(self.device).detach()
        old_logprobs = torch.stack(self.memory.logprobs).to(self.device).detach()
        old_actions_blind = torch.stack(self.memory.actions_blind).to(self.device).detach()
        old_logprobs_blind = torch.stack(self.memory.logprobs_blind).to(self.device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.k_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy, logprobs_blind, dist_blind_entropy = \
                self.policy.evaluate(old_states, old_actions, old_actions_blind)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp((logprobs - old_logprobs.detach() + logprobs_blind - old_logprobs_blind.detach()) / 2)

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.loss(state_values, rewards) \
                   - 0.01 * (dist_entropy + dist_blind_entropy) / 2

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

    def agent_start(self, state):
        action, blind_action = self.policy_old.act(state[0], self.memory)

        self.last_action = action
        self.last_state = state
        if self.discrete:
            action = self.action_space[action]
        blind_action = self.blind_action_space[blind_action]

        action_stpt, sat_sp = augment_ma(state, action)
        return action_stpt, sat_sp, blind_action

    def agent_step(self, reward, state):
        self.agent_end(reward, state)

        action_stpt, sat_sp, blind_action = self.agent_start(state)
        if state[1]["terminate"]:
            self.memory.clear_memory()
        return action_stpt, sat_sp, blind_action

    def agent_end(self, reward, state):
        self.memory.rewards.append(reward)
        self.memory.is_terminals.append(state[1]["terminate"])

        if state[2] % self.update_timestep == 0 or state[1]["terminate"]:
            self.update()
            self.memory.clear_memory()

    def save(self, num):
        self.policy.save_checkpoint(num)
        self.policy_old.save_checkpoint(num)
        self.memory.save(num)

    def load(self, num):
        self.policy.load_checkpoint(num)
        self.policy_old.load_checkpoint(num)
        self.memory.load(num)
