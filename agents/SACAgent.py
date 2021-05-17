import os, math, pdb, pickle
import numpy as np

import torch
import torch.nn.functional as F
from torch.optim import Adam

from agents.BaseAgent import BaseAgent2
from agents.ReplayMemory import ReplayMemory
from agents.networks.sac_networks import DeterministicPolicy, GaussianPolicy
from utils.for_agents import augment_ma, soft_update, hard_update


class SACAgent(BaseAgent2):
    def __init__(self, agent_info, Network, chkpt_dir='.'):

        super(SACAgent, self).__init__(agent_info, Network, chkpt_dir=chkpt_dir)

        self.type = 'SACAgent'

        # Create Action Space
        self.num_sat_actions = agent_info.get('num_sat_actions', 0)
        self.num_blind_actions = agent_info.get('num_blind_actions', 0)
        self.num_therm_actions = agent_info.get('num_therm_actions', 0)
        self.num_actions = self.num_sat_actions + self.num_blind_actions + self.num_therm_actions
        action_space = []
        for i in range(self.num_sat_actions):
            action_space.append([agent_info.get('min_sat_action'), agent_info.get('max_sat_action')])
        for i in range(self.num_therm_actions):
            action_space.append([agent_info.get('min_therm_action'), agent_info.get('max_therm_action')])
        for i in range(self.num_blind_actions):
            action_space.append([0, 180])
        self.action_space = np.array(action_space)
        self.control_blinds = self.num_blind_actions > 0

        # Set Hyperparameters
        self.updates = 0
        self.num_inputs = agent_info.get('n_state')
        self.hidden_size = agent_info.get('hidden_size')
        self.lr = agent_info.get('lr')
        self.batch_size = agent_info.get('batch_size')
        self.updates_per_step = agent_info.get('updates_per_step')
        self.plan_step = agent_info.get('plan_step')
        self.step = agent_info.get('step')
        self.forecasted = agent_info.get('forecasted')
        self.memory = ReplayMemory(agent_info.get('replay_size'), self.seed, chkpt_dir)
        self.start_steps = agent_info.get('start_steps')
        self.replay_size = agent_info.get('replay_size')
        self.gamma = agent_info.get('gamma')
        self.tau = agent_info.get('tau')
        self.alpha = agent_info.get('alpha')
        self.policy_type = agent_info.get('policy_type')
        self.target_update_interval = agent_info.get('target_update_interval')
        self.automatic_entropy_tuning = agent_info.get('automatic_entropy_tuning')
        self.device = torch.device("cuda" if agent_info.get('cuda') else "cpu")

        # Setup Networks
        self.critic = Network(self.num_inputs, self.num_actions, self.hidden_size,
                              'critic', chkpt_dir).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        self.critic_target = Network(self.num_inputs, self.num_actions, self.hidden_size,
                                     'critic_target', chkpt_dir).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            print('Using Gaussian')
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = (-1 * self.num_actions)
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=self.lr)

            self.policy = GaussianPolicy(
                self.num_inputs, self.num_actions, self.hidden_size,
                'policy', chkpt_dir,
                self.action_space,
            ).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=self.lr)

        else:
            print("Using Deterministic")
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(
                self.num_inputs, self.num_actions, self.hidden_size, self.action_space
            ).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=self.lr)

    def select_action(self, observation, evaluate=False):
        """

        :param observation:
        :param evaluate:
        :return: action is for the reward function
                 sat_sp is whats used by env_step and by the model for training
        """
        actions = []
        if self.start_steps > self.total_numsteps:
            for i in range(self.action_space.shape[0]):
                a = np.random.uniform(self.action_space[i].min(), self.action_space[i].max())
                actions.append(a)

        else:
            state, _, _ = observation
            state = torch.FloatTensor(state.float()).to(self.device).unsqueeze(0)
            if evaluate is False:
                action, _, _ = self.policy.sample(state)
            else:
                _, _, action = self.policy.sample(state)
            for i in range(self.action_space.shape[0]):
                a = action.detach().cpu().numpy()[0][i]
                actions.append(a)

        sat_actions = actions[0:self.num_sat_actions]
        therm_actions = actions[self.num_sat_actions:self.num_therm_actions + self.num_sat_actions]
        blind_actions = actions[self.num_therm_actions + self.num_sat_actions:]

        sat_actions_tups = []
        for a in sat_actions:
            action_stpt, sat_sp = augment_ma(observation, a)
            sat_actions_tups.append((action_stpt, sat_sp))
        if len(sat_actions) == 0:
            # this is hacky but makes the parsing in the main file cleaner
            sat_actions_tups.append(([], []))

        return sat_actions_tups, therm_actions, blind_actions, actions

    def learn(self):
        if len(self.memory) > self.batch_size:
            # Number of updates per step in environment
            for i in range(self.updates_per_step):
                self._update_parameters(self.memory, self.batch_size)
                self.updates += 1

    def _update_parameters(self, memory, batch_size):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch,
                               action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        if self.updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    def save(self, num):
        self.critic.save_checkpoint(num)
        self.critic_target.save_checkpoint(num)
        self.policy.save_checkpoint(num)
        self.memory.save(num)

    def load(self, num):
        self.critic.load_checkpoint(num)
        self.critic_target.load_checkpoint(num)
        self.policy.load_checkpoint(num)
        self.memory.load(num)
