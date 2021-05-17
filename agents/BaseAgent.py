import numpy as np
import torch


class BaseAgent:
    def __init__(self, *args, **kwargs):
        self.type = 'BaseAgent'

    def agent_start(self, *args, **kwargs):
        return None, None

    def agent_step(self, *args, **kwargs):
        return None, None

    def agent_end(self, *args, **kwargs):
        pass


class BaseAgent2:
    def __init__(self, agent_info, *args, **kwargs):
        # self.type = 'BaseAgent'
        self.total_numsteps = 0

        self.seed = agent_info.get('seed')
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def store_transition(self, observation, action, reward, observation_, done):
        state, _, _ = observation
        state_, _, _ = observation_
        self.memory.push(state, action, reward, state_, done)

    def agent_start(self, state):
        sat_actions, therm_actions, blind_actions, action_list = self.select_action(state)
        self.last_action = action_list
        self.last_state = state
        return sat_actions, therm_actions, blind_actions

    def agent_step(self, reward, state):
        self.total_numsteps += 1
        self.store_transition(self.last_state, self.last_action, reward, state, 1)
        self.learn()
        sat_actions, therm_actions, blind_actions, action_list = self.select_action(state)
        self.last_action = action_list
        self.last_state = state
        return sat_actions, therm_actions, blind_actions

    def agent_end(self, reward, state):
        self.total_numsteps += 1
        self.store_transition(self.last_state, self.last_action, reward, state, 1)
        self.learn()

    def inference_only(self, state):
        sat_actions, therm_actions, blind_actions, action_list = self.select_action(state)
        return sat_actions, therm_actions, blind_actions
