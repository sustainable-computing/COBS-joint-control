import torch
import torch.nn as nn
from torch.distributions import Categorical
import pickle
import os
from utils.for_agents import augment_ma
from numpy import linspace


class ReplayMemory:
    def __init__(self, action_num, chkpt_dir='.'):
        self.actions = [list() for _ in range(action_num)]
        self.states = list()
        self.logprobs = [list() for _ in range(action_num)]
        self.rewards = list()
        self.is_terminals = list()
        self.chkpt_dir = chkpt_dir

    def clear_memory(self):
        for action in self.actions:
            del action[:]
        for logprob in self.logprobs:
            del logprob[:]
        del self.states[:]
        del self.rewards[:]
        del self.is_terminals[:]

    def save(self, num):
        checkpoint_file = os.path.join(self.chkpt_dir, f'memory_{num}')
        dump_dict = {
            'actions': self.actions,
            'states': self.states,
            'logprobs': self.logprobs,
            'rewards': self.rewards,
            'is_terminals': self.is_terminals
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


class ActorCritic(nn.Module):
    def __init__(self, state_dim, Networks, device, name, action_space, chkpt_dir='.'):
        super(ActorCritic, self).__init__()
        self.device = device

        # actor
        self.action_network = Networks[0](action_space, f"ppo_actor_{name}", state_dim, chkpt_dir, Networks[2])

        # critic
        self.value_network = Networks[1](f"ppo_critic_{name}", state_dim, chkpt_dir)

        self.action_mean = None
        self.action_log_std = None
        self.action_scale = None
        self.action_bias = None
        self.action_space = action_space

    def inference_only(self, state):
        state = state.to(self.device)
        actions = self.action_network(state)

        action_scaled = list()
        for i, action_probs in enumerate(actions):
            dist = Categorical(action_probs)
            action = dist.sample()

            action_scaled.append(action.item())

        return action_scaled

    def act(self, state, memory):
        state = state.to(self.device)
        actions = self.action_network(state)
        # action_probs, blind_action_probs = self.action_network(state)

        action_scaled = list()
        for i, action_probs in enumerate(actions):
            dist = Categorical(action_probs)
            action = dist.sample()

            memory.actions[i].append(action)
            memory.logprobs[i].append(dist.log_prob(action))
            action_scaled.append(action.item())

        memory.states.append(state)

        return action_scaled

    def evaluate(self, state, eval_actions):
        actions = self.action_network(state)
        action_logprobs = list()
        dist_entropy = list()
        for i, action_probs in enumerate(actions):
            dist = Categorical(action_probs)
            action_logprobs.append(dist.log_prob(eval_actions[i]))
            dist_entropy.append(dist.entropy())

        state_value = self.value_network(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy

    def save_checkpoint(self, num):
        self.action_network.save_checkpoint(num)
        self.value_network.save_checkpoint(num)

    def load_checkpoint(self, num):
        self.action_network.load_checkpoint(num)
        self.value_network.load_checkpoint(num)


class PPOAgent:
    def __init__(self, agent_info, Networks, chkpt_dir='.'):
        # print(agent_info)
        self.type = "PPOAgent"
        torch.manual_seed(agent_info['seed'])
        self.lr = agent_info['lr']
        self.gamma = agent_info['gamma']
        self.eps_clip = agent_info['clip_param']
        self.update_timestep = agent_info['step']
        self.k_epochs = agent_info['update_episode']
        state_dim = agent_info['n_state']

        self.last_action = None
        self.last_state = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.num_sat_actions = agent_info.get('num_sat_actions', 0)
        self.num_blind_actions = agent_info.get('num_blind_actions', 0)
        self.num_therm_actions = agent_info.get('num_therm_actions', 0)
        self.num_actions = self.num_sat_actions + self.num_blind_actions + self.num_therm_actions
        self.action_space = list()
        for i in range(self.num_sat_actions):
            self.action_space.append(linspace(agent_info.get('min_sat_action'),
                                              agent_info.get('max_sat_action'),
                                              agent_info.get('num_possible_sat_setpoint')).round())
        for i in range(self.num_therm_actions):
            self.action_space.append(linspace(agent_info.get('min_therm_action'),
                                              agent_info.get('max_therm_action'),
                                              agent_info.get('num_possible_therm_setpoint')).round())
        for i in range(self.num_blind_actions):
            self.action_space.append(linspace(0, 180, agent_info.get('num_possible_blind_angle')).round())

        self.memory = ReplayMemory(self.num_actions, chkpt_dir)

        self.policy = ActorCritic(state_dim, Networks, self.device,
                                  "policy", self.action_space, chkpt_dir).to(self.device)
        self.policy_old = ActorCritic(state_dim, Networks, self.device,
                                      "old_policy", self.action_space, chkpt_dir).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = torch.optim.RMSprop(self.policy.parameters(), lr=self.lr)
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
        old_actions = list()
        old_logprobs = list()
        for i in range(len(self.memory.actions)):
            old_actions.append(torch.stack(self.memory.actions[i]).to(self.device).detach())
            old_logprobs.append(torch.stack(self.memory.logprobs[i]).to(self.device).detach())

        # Optimize policy for K epochs:
        for _ in range(self.k_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            value = logprobs[0] - old_logprobs[0].detach()
            for i in range(1, len(logprobs)):
                value += logprobs[i] - old_logprobs[i].detach()
            ratios = torch.exp(value / len(logprobs))

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            entropy = dist_entropy[0]
            for e in dist_entropy:
                entropy += e
            loss = -torch.min(surr1, surr2) + 0.5 * self.loss(state_values, rewards) \
                   - 0.01 * (entropy / len(dist_entropy))

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

    def inference_only(self, state):
        self.memory.clear_memory()
        action_idx = self.policy.inference_only(state[0])

        self.last_action = action_idx[:]
        self.last_state = state
        actions = list()
        for i, action in enumerate(action_idx):
            actions.append(self.action_space[i][action])

        sat_actions = actions[:self.num_sat_actions]
        therm_actions = actions[self.num_sat_actions:self.num_therm_actions + self.num_sat_actions]
        blind_actions = actions[self.num_therm_actions + self.num_sat_actions:]

        sat_actions_tups = []
        for action in sat_actions:
            action_stpt, sat_sp = augment_ma(state, action)
            sat_actions_tups.append((action_stpt, sat_sp))
        if len(sat_actions) == 0:
            # this is hacky but makes the parsing in the main file cleaner
            sat_actions_tups.append(([], []))

        return sat_actions_tups, therm_actions, blind_actions

    def agent_start(self, state):
        action_idx = self.policy_old.act(state[0], self.memory)

        self.last_action = action_idx[:]
        self.last_state = state
        actions = list()
        for i, action in enumerate(action_idx):
            actions.append(self.action_space[i][action])

        sat_actions = actions[:self.num_sat_actions]
        therm_actions = actions[self.num_sat_actions:self.num_therm_actions + self.num_sat_actions]
        blind_actions = actions[self.num_therm_actions + self.num_sat_actions:]

        sat_actions_tups = []
        for action in sat_actions:
            action_stpt, sat_sp = augment_ma(state, action)
            sat_actions_tups.append((action_stpt, sat_sp))
        if len(sat_actions) == 0:
            # this is hacky but makes the parsing in the main file cleaner
            sat_actions_tups.append(([], []))

        return sat_actions_tups, therm_actions, blind_actions

    def agent_step(self, reward, state):
        self.agent_end(reward, state)

        sat_actions_tups, therm_actions, blind_actions = self.agent_start(state)
        if state[1]["terminate"]:
            self.memory.clear_memory()
        return sat_actions_tups, therm_actions, blind_actions

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
