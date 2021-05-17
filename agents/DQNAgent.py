import pdb, os
import numpy as np
import torch as T


from agents.BaseAgent import BaseAgent
from agents.ReplayMemory import ReplayMemory
from utils.for_agents import augment_ma


class DQNAgent(BaseAgent):

    def __init__(self, agent_info, Network, chkpt_dir='.'):
        self.type = 'DQNAgent'
        self.gamma = agent_info.get('gamma')
        self.epsilon = agent_info.get('epsilon')
        self.lr = agent_info.get('lr')
        self.input_dims = agent_info.get('input_dims')
        self.batch_size = agent_info.get('batch_size')
        self.eps_min = agent_info.get('eps_min')
        self.eps_dec = agent_info.get('eps_dec')
        self.replace_target_cnt = agent_info.get('replace')
        self.min_action = agent_info.get('min_action')
        self.max_action = agent_info.get('max_action')
        self.num_actions = agent_info.get('num_actions')

        self.action_space = np.linspace(self.min_action, self.max_action, self.num_actions).round()
        # rnge = (self.max_action - self.min_action) / self.num_actions
        # self.action_space = [i for i in range(self.min_action, self.max_action, int(rnge))]
        # assert len(self.action_space) == self.num_actions

        self.chkpt_dir = chkpt_dir
        self.learn_step_counter = 0

        self.memory = ReplayMemory(
            agent_info.get('mem_size'),
            agent_info.get('seed'),
            chkpt_dir
        )

        self.q_eval = Network(
            self.lr, self.num_actions,
            input_dims=self.input_dims,
            name='q_eval',
            chkpt_dir=self.chkpt_dir)

        self.q_next = Network(
            self.lr, self.num_actions,
            input_dims=self.input_dims,
            name='q_next',
            chkpt_dir=self.chkpt_dir)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state, _, _ = observation

            actions = self.q_eval.forward(state)
            action_idx = T.argmax(actions).item()
            action = self.action_space[action_idx]

        else:
            action = np.random.choice(self.action_space)

        action, sat_sp = augment_ma(observation, action)

        return action, sat_sp

    def store_transition(self, observation, action, reward, observation_, done):
        # TODO - need to test that the action index is working properly
        action, sat_sp = action
        action_idx = np.where(self.action_space == action)
        # action_idx = self.action_space.index(action)
        # state = self.create_state_with_forecasted(observation)
        # state_ = self.create_state_with_forecasted(observation_)
        state, _, _ = observation
        state_, _, _ = observation_
        self.memory.push(state, action_idx, reward, state_, done)

    def sample_memory(self):
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

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()
        indices = np.arange(self.batch_size)

        # TODO check shape of actions (see DuelingDQN)
        q_pred = self.q_eval.forward(states)[indices, actions].double()
        q_next = self.q_next.forward(states_).max(dim=1)[0]

        dones = T.ByteTensor(dones)
        q_next[dones] = 0.0

        q_target = rewards + self.gamma * q_next.double()

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()

    def agent_start(self, state):
        action = self.choose_action(state)
        self.last_action = action
        self.last_state = state
        return self.last_action

    def agent_step(self, reward, state):
        self.store_transition(self.last_state, self.last_action, reward, state, False)
        self.learn()
        action = self.choose_action(state)
        self.last_action = action
        self.last_state = state
        return self.last_action

    def agent_end(self, reward, state):
        self.store_transition(self.last_state, self.last_action, reward, state, False)
        self.learn()

    def save(self, num):
        self.q_eval.save_checkpoint(num)
        self.q_next.save_checkpoint(num)
        self.memory.save(num)

    def load(self, num):
        self.q_eval.load_checkpoint(num)
        self.q_next.load_checkpoint(num)
        self.memory.load(num)
