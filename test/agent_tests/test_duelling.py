import pdb
import unittest

import torch
import pandas as pd
import numpy as np

from agents.BranchingDuelingDQNAgent import BranchingDuelingDQNAgent
from cobs.model import Model
from test.test_config import state_name, eplus_naming_dict, eplus_var_types, \
    SatAction, BlindActionSingleZone, BlindActionMultiZone, branching_dueling_dqn_network_map

from utils.rewards import ViolationPActionReward


class DQNTest(unittest.TestCase):

    agent_params = {
        "lr": 0.0001,
        "mem_size": 200,
        "gamma": 0.99,
        "batch_size": 64,
        "eps_min": 0.1,
        "replace": 1000,
        "eps_dec": 0.001,
        'discrete_sat_actions': 66,
        'discrete_blind_actions': 33,
        'min_action': -20,
        'min_sat_action': -20,
        'max_action': 20,
        'max_sat_action': 20,
        'num_actions': 66,
        'input_dims': (len(state_name),),
        'epsilon': 0.1,
        'seed': 42
    }
    eplus_path = '/Applications/EnergyPlus-9-3-0-bugfix/'
    epw_path = 'test/eplus_files/test.epw'
    Model.set_energyplus_folder(eplus_path)

    @staticmethod
    def run_episode(ep_model, agent, blinds_zone_multi=False):
        if blinds_zone_multi:
            BlindAction = BlindActionMultiZone
        else:
            BlindAction = BlindActionSingleZone

        observations = []
        sat_actions_list = []
        blind_actions_list = []
        therm_actions_list = []

        obs = ep_model.reset()
        observations.append(obs)

        state = torch.tensor([obs[name] for name in state_name]).double()
        action = agent.agent_start((state, obs, 0))

        sat_actions, therm_actions, blind_actions = action
        sat_actions_list.append(sat_actions[0])
        therm_actions_list.append(therm_actions)
        blind_actions_list.append(blind_actions)

        while not ep_model.is_terminate():
            # # SETUP ACTIONS
            env_actions = []
            stpt_actions = SatAction(sat_actions[0][1], obs, sat_actions[0][0])

            blind_actions = BlindAction(blind_actions, obs)

            env_actions += stpt_actions
            env_actions += blind_actions

            obs = ep_model.step(env_actions)
            observations.append(obs)
            state = torch.tensor([obs[name] for name in state_name]).double()
            feeding_state = (state, obs, obs["timestep"])
            action = agent.agent_step(obs["reward"], feeding_state)

            sat_actions, therm_actions, blind_actions = action
            sat_actions_list.append(sat_actions[0])
            therm_actions_list.append(therm_actions)
            blind_actions_list.append(blind_actions)

        return observations, (sat_actions_list, therm_actions_list,  blind_actions_list), agent

    def setup_env(self, idf_path):
        reward = ViolationPActionReward(1)

        ep_model = Model(
            idf_file_name=idf_path,
            weather_file=self.epw_path,
            eplus_naming_dict=eplus_naming_dict,
            eplus_var_types=eplus_var_types,
            reward=reward,
            tmp_idf_path='test/agent_tests/test_results'
        )
        ep_model.set_runperiod(*(1, 1991, 1, 1))

        return ep_model

    def test_duelling_sat(self):
        self.agent_params["num_sat_actions"] = 1
        self.agent_params["num_blind_actions"] = 0
        self.agent_params["num_therm_actions"] = 0

        network = branching_dueling_dqn_network_map['octo']
        agent = BranchingDuelingDQNAgent(self.agent_params, network, chkpt_dir='test/agent_tests/test_results')

        ep_model = self.setup_env('test/eplus_files/test_control.idf')

        observations, actions, agent = self.run_episode(ep_model, agent)
        obs_test = pd.DataFrame.from_dict(observations)

        sat_actions, therm_actions, blind_actions = actions

        obs_test['actions'] = [a1 for a1, _ in sat_actions]
        obs_test['sat_stpts'] = [a2.item() for _, a2 in sat_actions]

        float_cols = [
            'Outdoor Temp.',
            'Diff. Solar Rad.',
            'Direct Solar Rad.',
            'Indoor Temp.',
            'Indoor Temp. Setpoint',
            'PPD',
            'Occupancy Flag',
            'Blind Angle Zone 1',
            'Heat Coil Power',
            'HVAC Power',
            'Sys Out Temp.',
            'MA Temp.',
            'actions',
            'sat_stpts',
        ]

        obs_true = pd.read_csv('test/agent_tests/saved_results/duelling_no_blinds_obs.csv')
        for c in float_cols:
            close = np.isclose(obs_test[c], obs_true[c])
            self.assertTrue(close.all())

    def test_duelling_blinds_one_zone(self):
        self.agent_params["num_sat_actions"] = 1
        self.agent_params["num_blind_actions"] = 1
        self.agent_params["num_therm_actions"] = 0

        # self.agent_params['num_stpt_actions'] = 66
        # self.agent_params['num_blind_actions'] = 33

        network = branching_dueling_dqn_network_map['octo']
        agent = BranchingDuelingDQNAgent(self.agent_params, network, chkpt_dir='test/agent_tests/test_results')

        ep_model = self.setup_env('test/eplus_files/test_control.idf')
        ep_model.set_blinds(
            ["WF-1"],
            blind_material_name="White Painted Metal Blind",
            agent_control=True
        )

        observations, actions, agent = self.run_episode(ep_model, agent)
        obs_test = pd.DataFrame.from_dict(observations)

        sat_actions, therm_actions, blind_actions = actions

        obs_test['actions'] = [a1 for a1, _ in sat_actions]
        obs_test['sat_stpts'] = [a2.item() for _, a2 in sat_actions]
        obs_test['blind_actions'] = [a1[0] for a1 in blind_actions]

        # obs_test.to_csv('test/agent_tests/saved_results/duelling_blinds_obs.csv', index=False)

        float_cols = [
            'Outdoor Temp.',
            'Diff. Solar Rad.',
            'Direct Solar Rad.',
            'Indoor Temp.',
            'Indoor Temp. Setpoint',
            'PPD',
            'Occupancy Flag',
            'Blind Angle Zone 1',
            'Heat Coil Power',
            'HVAC Power',
            'Sys Out Temp.',
            'MA Temp.',
            'actions',
            'sat_stpts',
            'blind_actions'
        ]

        obs_true = pd.read_csv('test/agent_tests/saved_results/duelling_blinds_obs.csv')
        for c in float_cols:
            close = np.isclose(obs_test[c], obs_true[c])
            self.assertTrue(close.all())

    def test_duelling_blinds_many_zone_many_setpoint(self):
        self.agent_params["num_sat_actions"] = 1
        self.agent_params["num_blind_actions"] = 4
        self.agent_params["num_therm_actions"] = 0

        # self.agent_params['num_stpt_actions'] = 66
        # self.agent_params['num_blind_actions'] = 33

        network = branching_dueling_dqn_network_map['octo']
        agent = BranchingDuelingDQNAgent(self.agent_params, network, chkpt_dir='test/agent_tests/test_results')

        ep_model = self.setup_env('test/eplus_files/5Zone_Control_SAT_no_windowcontrol.idf')
        ep_model.set_blinds(
            ["WF-1", "WR-1", "WB-1", "WL-1"],
            blind_material_name="White Painted Metal Blind",
            agent_control=True
        )

        observations, actions, agent = self.run_episode(ep_model, agent, blinds_zone_multi=True)
        obs_test = pd.DataFrame.from_dict(observations)

        sat_actions, therm_actions, blind_actions = actions

        obs_test['actions'] = [a1 for a1, _ in sat_actions]
        obs_test['sat_stpts'] = [a2.item() for _, a2 in sat_actions]
        for i in range(0, len(blind_actions[0])):
            obs_test[f'Blind Action {i+1}'] = [a1[i] for a1 in blind_actions]

        # obs_test.to_csv('test/agent_tests/saved_results/duelling_blinds_multi_zone_multi_stpt_obs.csv', index=False)

        float_cols = [
            'Outdoor Temp.',
            'Diff. Solar Rad.',
            'Direct Solar Rad.',
            'Indoor Temp.',
            'Indoor Temp. Setpoint',
            'PPD',
            'Occupancy Flag',
            'Blind Angle Zone 1',
            'Heat Coil Power',
            'HVAC Power',
            'Sys Out Temp.',
            'MA Temp.',
            'actions',
            'sat_stpts',
            'Blind Action 1',
            'Blind Action 2',
            'Blind Action 3',
            'Blind Action 4'
        ]
        obs_true = pd.read_csv('test/agent_tests/saved_results/duelling_blinds_multi_zone_multi_stpt_obs.csv')
        for c in float_cols:
            close = np.isclose(obs_test[c], obs_true[c])
            self.assertTrue(close.all())


if __name__ == '__main__':
    unittest.main()
