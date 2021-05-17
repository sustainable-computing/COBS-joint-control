import pdb, os

import torch
import numpy as np
import torch as T
import torch.nn as nn

from agents.BaseAgent import BaseAgent
from agents.ReplayMemory import ReplayMemory
from utils.for_agents import augment_ma


class DQNAgentBlinds(BaseAgent):

    def __init__(self, agent_info, Network, chkpt_dir='.'):
        self.type = 'DQNAgent'
        self.gamma = agent_info.get('gamma')
        self.epsilon = agent_info.get('epsilon')
        self.lr = agent_info.get('lr')
        self.input_dims = agent_info.get('input_dims')
        self.batch_size = agent_info.get('batch_size')
        self.step = agent_info.get('step')
        self.eps_min = agent_info.get('eps_min')
        self.eps_dec = agent_info.get('eps_dec')
        self.replace_target_cnt = agent_info.get('replace')
        self.min_action = agent_info.get('min_action')
        self.max_action = agent_info.get('max_action')
        self.num_stpt_actions = agent_info.get('num_stpt_actions')
        self.num_blind_actions = agent_info.get('num_blind_actions')
        self.hidden_size = agent_info.get('hidden_size')

        stpt_action_space = np.linspace(self.min_action, self.max_action, self.num_stpt_actions).round()
        blind_action_space = np.linspace(0, 180, self.num_blind_actions).round()

        self.action_space = np.array([stpt_action_space, blind_action_space])

        self.chkpt_dir = chkpt_dir
        self.learn_step_counter = 0

        self.memory = ReplayMemory(
            agent_info.get('mem_size'),
            agent_info.get('seed'),
            chkpt_dir
        )

        self.q_eval = Network(
            self.lr, self.num_stpt_actions,
            self.num_blind_actions, self.hidden_size,
            input_dims=self.input_dims,
            name='q_eval',
            chkpt_dir=self.chkpt_dir)

        self.q_next = Network(
            self.lr, self.num_stpt_actions,
            self.num_blind_actions, self.hidden_size,
            input_dims=self.input_dims,
            name='q_next',
            chkpt_dir=self.chkpt_dir)

    def select_action(self, observation):
        if np.random.random() > self.epsilon:
            state, _, _ = observation

            actions_stpt, actions_blinds = self.q_eval.forward(state)

            # setpoint actions
            actions_stpt_idx = T.argmax(actions_stpt).item()
            action_stpt = self.action_space[0][actions_stpt_idx]

            # blind actions
            actions_blinds_idx = T.argmax(actions_blinds).item()
            action_blinds = self.action_space[1][actions_blinds_idx]

        else:
            action_stpt = np.random.choice(self.action_space[0])
            action_blinds = np.random.choice(self.action_space[1])

        action_stpt, sat_sp = augment_ma(observation, action_stpt)

        return action_stpt, sat_sp, action_blinds

    def store_transition(self, observation, action, reward, observation_, done):
        # TODO - need to test that the action index is working properly
        action_stpt, action_blind = action
        action_stpt_idx = np.where(self.action_space[0] == action_stpt)
        action_blind_idx = np.where(self.action_space[1] == action_blind)

        state, _, _ = observation
        state_, _, _ = observation_
        self.memory.push(state, [action_stpt_idx, action_blind_idx], reward, state_, done)

    def sample_memory(self):
        # state, action, reward, new_state, done = \
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.memory.sample(
            batch_size=self.batch_size)

        states = T.tensor(state_batch).to(self.q_eval.device)
        actions = T.tensor(action_batch).to(self.q_eval.device)
        rewards = T.tensor(reward_batch).to(self.q_eval.device)
        dones = T.ByteTensor(mask_batch.astype(int)).to(self.q_eval.device)
        states_ = T.tensor(next_state_batch).to(self.q_eval.device)

        return states, actions, rewards, states_, dones

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min

    def save(self, num):
        self.q_eval.save_checkpoint(num)
        self.q_next.save_checkpoint(num)
        self.memory.save(num)

    def load(self, num):
        self.q_eval.load_checkpoint(num)
        self.q_next.load_checkpoint(num)
        self.memory.load(num)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()
        indices = np.arange(self.batch_size)

        actions_stpt = actions[:, 0, 0, 0]
        actions_blinds = actions[:, 1, 0, 0]

        q_pred_stpt, q_pred_blinds = self.q_eval.forward(states)

        q_pred_stpt = q_pred_stpt[indices, actions_stpt].double()
        q_pred_blinds = q_pred_blinds[indices, actions_blinds].double()

        q_next_stpt, q_next_blinds = self.q_next.forward(states_)

        q_next_stpt = q_next_stpt.max(dim=1)[0]
        q_next_blinds = q_next_blinds.max(dim=1)[0]

        dones = T.ByteTensor(dones)
        q_next_stpt[dones] = 0.0
        q_next_blinds[dones] = 0.0

        q_stpt_target = rewards + self.gamma * q_next_stpt.double()
        q_blinds_target = rewards + self.gamma * q_next_blinds.double()

        loss_stpt = self.q_eval.loss(q_stpt_target, q_pred_stpt).to(self.q_eval.device)
        loss_blinds = self.q_eval.loss(q_blinds_target, q_pred_blinds).to(self.q_eval.device)
        torch.autograd.backward([loss_stpt, loss_blinds])

        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()

    def agent_start(self, state):
        action = self.select_action(state)
        self.last_action = [action[0], action[2]]
        self.last_state = state
        return action

    def agent_step(self, reward, state):
        self.store_transition(self.last_state, self.last_action, reward, state, 1)
        self.learn()
        action = self.select_action(state)
        self.last_action = [action[0], action[2]]
        self.last_state = state
        return action

    def agent_end(self, reward, state):
        self.store_transition(self.last_state, self.last_action, reward, state, 1)
        self.learn()
