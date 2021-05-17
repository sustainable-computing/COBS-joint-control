import numpy as np
import torch as T

from agents.BaseAgent import BaseAgent
from agents.ReplayMemory import ReplayMemory
from utils.for_agents import augment_ma


class DDQNAgent(BaseAgent):
    # def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
    #              mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
    #              replace=1000, algo=None, env_name=None, chkpt_dir='tmp/dqn'):

    def __init__(self, agent_info, Network, chkpt_dir='.'):
        self.type = 'DDQNAgent'
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
            # state = T.tensor([observation],dtype=T.float).to(self.q_eval.device)
            # actions = self.q_eval.forward(state)
            # action = T.argmax(actions).item()

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

    # def store_transition(self, state, action, reward, state_, done):
        # self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(self):
        state, action, reward, new_state, done = \
                                self.memory.sample(batch_size=self.batch_size)

        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)

        return states, actions, rewards, states_, dones

    def replace_target_network(self):
        if self.replace_target_cnt is not None and \
           self.learn_step_counter % self.replace_target_cnt == 0:
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

        q_pred = self.q_eval.forward(states)[indices, actions]
        q_next = self.q_next.forward(states_)
        q_eval = self.q_eval.forward(states_)

        max_actions = T.argmax(q_eval, dim=1)
        q_next[dones] = 0.0

        q_target = rewards + self.gamma*q_next[indices, max_actions]
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
