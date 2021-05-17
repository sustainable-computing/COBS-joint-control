import pdb

import numpy as np
import torch as T
from random import sample

from agents.BaseAgent import BaseAgent2
from agents.ReplayMemory import ReplayMemory
from utils.for_agents import augment_ma
from itertools import combinations_with_replacement, product

class Dummy(BaseAgent2):

    def __init__(self, agent_info, network, chkpt_dir='.'):

        super(Dummy, self).__init__(agent_info)

        self.type = 'Dummy'

        # Create action space
        self.num_sat_actions = agent_info.get('num_sat_actions', 0)
        self.num_blind_actions = agent_info.get('num_blind_actions', 0)
        self.num_therm_actions = agent_info.get('num_therm_actions', 0)

        self.discrete_sat_actions = 20
        self.discrete_blind_actions = agent_info.get('discrete_blind_actions', 0)
        self.discrete_therm_actions = agent_info.get('discrete_therm_actions', 0)

        self.stpt_action_space = np.linspace(
                agent_info.get('min_sat_action'), agent_info.get('max_sat_action'), self.discrete_sat_actions).round()
        self.therm_action_space = np.linspace(
            agent_info.get('min_therm_action', 0), agent_info.get('max_therm_action', 40), self.discrete_therm_actions).round()
        self.blind_action_space = np.linspace(0, 180, self.discrete_blind_actions).round()

        self.memory = ReplayMemory(
            100000,
            self.seed,
            chkpt_dir
        )

    def select_action(self, observation):
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

        for i in range(len(idxs)):
            observation[1][f"Action idx {i}"] = idxs[i]

        return sat_actions_tups, therm_actions, blind_actions, idxs

    def learn(self,plan=False):
        pass

    def save(self, num):
        self.memory.save(num)