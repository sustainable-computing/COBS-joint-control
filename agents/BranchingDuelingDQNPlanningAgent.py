import pdb

import numpy as np
import torch as T
from random import sample

from agents.BaseAgent import BaseAgent2
from agents.ReplayMemory import ReplayMemory
from utils.for_agents import augment_ma
from itertools import combinations_with_replacement, product

class BranchingDuelingDQNPlanningAgent(BaseAgent2):

    def __init__(self, agent_info, Network, WorldNetwork=None ,chkpt_dir='.'):

        super(BranchingDuelingDQNPlanningAgent, self).__init__(agent_info, Network, chkpt_dir=chkpt_dir)

        self.type = 'BDQNwPlanningAgent'


        # Create action space
        self.num_sat_actions = agent_info.get('num_sat_actions', 0)
        self.num_blind_actions = agent_info.get('num_blind_actions', 0)
        self.num_therm_actions = agent_info.get('num_therm_actions', 0)

        self.discrete_sat_actions = agent_info.get('discrete_sat_actions', 0)
        self.discrete_blind_actions = agent_info.get('discrete_blind_actions', 0)
        self.discrete_therm_actions = agent_info.get('discrete_therm_actions', 0)

        self.stpt_action_space = np.linspace(
                agent_info.get('min_sat_action'), agent_info.get('max_sat_action'), self.discrete_sat_actions).round()
        self.therm_action_space = np.linspace(
            agent_info.get('min_therm_action', 0), agent_info.get('max_therm_action', 40), self.discrete_therm_actions).round()
        self.blind_action_space = np.linspace(0, 180, self.discrete_blind_actions).round()

        # Set Hyperparameters
        self.gamma = agent_info.get('gamma')
        self.epsilon = agent_info.get('epsilon')
        self.lr = agent_info.get('lr')
        self.input_dims = agent_info.get('input_dims')
        self.batch_size = agent_info.get('batch_size')
        self.eps_min = agent_info.get('eps_min')
        self.eps_dec = agent_info.get('eps_dec')
        self.replace_target_cnt = agent_info.get('replace')
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.planning_data = None
        self.real_states = None
        self.real_actions = None
        self.sim_rewards = None
        self.sim_states_ = None
        self.num_samples = 20

        self.chkpt_dir = chkpt_dir
        self.learn_step_counter = 0

        self.memory = ReplayMemory(
            agent_info.get('mem_size'),
            self.seed,
            chkpt_dir
        )

        self.q_eval = Network(
            self.lr,
            self.num_sat_actions, self.num_therm_actions, self.num_blind_actions,
            self.discrete_sat_actions, self.discrete_therm_actions, self.discrete_blind_actions,
            input_dims=self.input_dims,
            name='q_eval',
            chkpt_dir=self.chkpt_dir)

        self.world_network = WorldNetwork(
            self.lr,
            self.num_sat_actions, self.num_therm_actions, self.num_blind_actions,
            self.discrete_sat_actions, self.discrete_therm_actions, self.discrete_blind_actions,
            input_dims=self.input_dims,
            name='world',
            chkpt_dir=self.chkpt_dir)

        self.q_next = Network(
            self.lr,
            self.num_sat_actions, self.num_therm_actions, self.num_blind_actions,
            self.discrete_sat_actions, self.discrete_therm_actions, self.discrete_blind_actions,
            input_dims=self.input_dims,
            name='q_next',
            chkpt_dir=self.chkpt_dir)

    def select_action(self, observation):
        if np.random.random() > self.epsilon:
            state, _, _ = observation

            _, a_stpts, a_therms, a_blinds = self.q_eval.forward(state.to(self.device))
            stpt_actions = []
            therm_actions = []
            blind_actions = []
            idxs = []

            for a in a_stpts:
                actions_stpt_idx = T.argmax(a).item()
                stpt_actions.append(self.stpt_action_space[actions_stpt_idx])
                idxs.append(actions_stpt_idx)
            for a in a_therms:
                actions_stpt_idx = T.argmax(a).item()
                therm_actions.append(self.therm_action_space[actions_stpt_idx])
                idxs.append(actions_stpt_idx)
            for a in a_blinds:
                actions_stpt_idx = T.argmax(a).item()
                blind_actions.append(self.blind_action_space[actions_stpt_idx])
                idxs.append(actions_stpt_idx)
        else:
            stpt_actions = []
            therm_actions = []
            blind_actions = []
            idxs = []

            for i in range(0, self.num_sat_actions):
                rand = np.random.choice(self.stpt_action_space)
                stpt_actions.append(rand)
                idxs.append(np.where(self.stpt_action_space == rand)[0].item())
            for i in range(0, self.num_therm_actions):
                rand = np.random.choice(self.therm_action_space)
                therm_actions.append(rand)
                idxs.append(np.where(self.therm_action_space == rand)[0].item())
            for i in range(0, self.num_blind_actions):
                rand = np.random.choice(self.blind_action_space)
                blind_actions.append(rand)
                idxs.append(np.where(self.blind_action_space == rand)[0].item())

        sat_actions_tups = []
        for a in stpt_actions:
            action_stpt, sat_sp = augment_ma(observation, a)
            sat_actions_tups.append((action_stpt, sat_sp))
        return sat_actions_tups, therm_actions, blind_actions, idxs

    def sample_memory(self):
        state, action, reward, new_state, done = \
            self.memory.sample(batch_size=self.batch_size)

        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        dones = T.ByteTensor(done.astype(int)).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)

        return states, actions, rewards, states_, dones

    def replace_target_network(self):
        if self.replace_target_cnt is not None and \
           self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min

    def get_world_rmse(self, obs):
        print(self.last_state)
        print(self.last_action)
        print(obs)

    def learn(self,plan=False):
        if len(self.memory) < self.batch_size:
            return
        self.q_eval.optimizer.zero_grad()
        self.replace_target_network()

        if plan:
            states, actions, rewards, states_ = self.get_planning_data()
            indices = np.arange(self.num_samples)
        else:
            states, actions, rewards, states_, dones= self.sample_memory()
            indices = np.arange(self.batch_size)


        V_s, A_stpts, A_therms, A_blinds = self.q_eval.forward(states)
        V_s_, A_stpts_, A_therms_, A_blinds_ = self.q_next.forward(states_)

        actions = actions.type(T.long)

        losses = []
        act_idx = 0
        for A_s, A_s_ in zip(A_stpts, A_stpts_):
            q_pred = T.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions[:, act_idx]].double()
            q_next = T.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True))).max(dim=1)[0]
            q_target = rewards + self.gamma*q_next.double()
            loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
            losses.append(loss)
            act_idx += 1

        for A_s, A_s_ in zip(A_therms, A_therms_):
            q_pred = T.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions[:, act_idx]].double()
            q_next = T.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True))).max(dim=1)[0]
            q_target = rewards + self.gamma*q_next.double()
            loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
            losses.append(loss)
            act_idx += 1

        for A_s, A_s_ in zip(A_blinds, A_blinds_):
            q_pred = T.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions[:, act_idx]].double()
            q_next = T.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True))).max(dim=1)[0]
            q_target = rewards + self.gamma*q_next.double()
            loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
            losses.append(loss)
            act_idx += 1

        T.autograd.backward(losses)
        self.q_eval.optimizer.step()

        if not plan:
            self.train_model_world()
            self.learn_step_counter += 1
            self.decrement_epsilon()

    def train_model_world(self):
        self.world_network.optimizer.zero_grad()

        states, actions, rewards, states_, dones = self.sample_memory()
        model_input = T.cat((states,actions),1)
        model_output = T.cat((states_,T.transpose(rewards.unsqueeze(0),0,1)),1)

        prediction = self.world_network.forward(model_input.float())
        loss_v = self.world_network.loss(prediction,model_output.float())
        T.autograd.backward(loss_v)
        self.world_network.optimizer.step()

    def plan(self):

        if len(self.memory) < self.batch_size:
            return
        action_size = self.num_sat_actions + self.num_therm_actions + self.num_blind_actions,
        actions_space = np.arange(0, 20)
        actions_combinations = list(product(actions_space, repeat=action_size[0]))
        actions_combinations = sample(actions_combinations, self.num_samples)
        actions_combinations = T.FloatTensor(actions_combinations)

        states, _, _, _, _ = self.sample_memory()

        for state in states:
            states_repeat = state.repeat(self.num_samples,1)

            model_input = T.cat((states_repeat, actions_combinations), 1)
            model_output_sim = self.world_network.forward(model_input.float())

            sim_states = model_output_sim[:,:-1]
            sim_rewards = model_output_sim[:,-1]

            self.set_planning_data(states_repeat,actions_combinations,sim_rewards, sim_states)
            self.learn(plan=True)

    def set_planning_data(self,states,actions,rewards,states_):
        self.real_states = states
        self.real_actions = actions
        self.sim_rewards = rewards
        self.sim_states_ = states_

    def get_planning_data(self):
        return self.real_states, self.real_actions, self.sim_rewards, self.sim_states_

    def save(self, num):
        self.q_eval.save_checkpoint(num)
        self.q_next.save_checkpoint(num)
        self.memory.save(num)

    def load(self, num):
        self.q_eval.load_checkpoint(num)
        self.q_next.load_checkpoint(num)
        self.memory.load(num)
