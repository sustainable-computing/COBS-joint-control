import pdb
import unittest

import torch
import pandas as pd
import numpy as np

from agents.SACAgent import SACAgent
from cobs.model import Model
from test.test_config import state_name, sac_network_map, eplus_naming_dict, eplus_var_types, \
                             SatAction, BlindActionSingleZone, ThermActionSingleZone, BlindActionMultiZone,\
                             ThermActionMultiZone

from utils.rewards import ViolationPActionReward


# SatAction = ActionCreator("Schedule:Constant", "Schedule Value", "SAT_SP")
# BlindActionSingle = ActionCreator("Schedule:Constant", "Schedule Value", "WF-1_shading_schedule")
# ThermActionSingle = ActionCreator("Zone Temperature Control", "Heating Setpoint", "SPACE1-1")


class SACTest(unittest.TestCase):

    agent_params = {
        "policy_type": "Gaussian",
        "gamma": 0.99,
        "tau": 0.005,
        "lr": 0.0003,
        "batch_size": 2,
        "hidden_size": 2,
        "updates_per_step": 1,
        "target_update_interval": 1,
        "replay_size": 200,
        "cuda": False,
        "step": 300 * 3,
        "start_steps": 5,
        "alpha": 0.2,
        "automatic_entropy_tuning": False,
        "num_inputs": len(state_name),
        "min_sat_action": -20,
        "max_sat_action": 20,
        "seed": 42
    }
    eplus_path = '/Applications/EnergyPlus-9-' \
                 '3-0-bugfix/'
    # idf_path = 'test/eplus_files/test_control.idf'
    # idf_path = 'test/eplus_files/5Zone_Control_SAT.idf'
    epw_path = 'test/eplus_files/test.epw'
    Model.set_energyplus_folder(eplus_path)

    def test_sac_sat(self):
        self.agent_params["num_sat_actions"] = 1
        self.agent_params["num_blind_actions"] = 0
        self.agent_params["num_therm_actions"] = 0

        network = sac_network_map['leaky']
        agent = SACAgent(self.agent_params, network, chkpt_dir='test/agent_tests/test_results')

        ep_model = self.setup_env('test/eplus_files/test_control.idf')

        observations, actions, agent = self.run_episode(ep_model, agent, "SAT_SP")
        obs_test = pd.DataFrame.from_dict(observations)

        sat_actions, therm_actions, blind_actions = actions

        # pdb.set_trace()

        obs_test['actions'] = [a1 for a1, _ in sat_actions]
        obs_test['sat_stpts'] = [a2.item() for _, a2 in sat_actions]

        # obs_test['blind_actions'] = blind_actions

        float_cols = [
            'Outdoor Temp.',
            'Diff. Solar Rad.',
            'Direct Solar Rad.',
            'Indoor Temp.',
            'Indoor Temp. Setpoint',
            'PPD',
            'Occupancy Flag',
            'Heat Coil Power',
            'HVAC Power',
            'Sys Out Temp.',
            'MA Temp.',
            'actions',
            'sat_stpts'
        ]

        obs_true = pd.read_csv('test/agent_tests/saved_results/sac_no_blinds_obs.csv')
        for c in float_cols:
            close = np.isclose(obs_test[c], obs_true[c])
            self.assertTrue(close.all())

    def test_sac_blinds_one_zone(self):
        self.agent_params["num_sat_actions"] = 1
        self.agent_params["num_blind_actions"] = 1
        self.agent_params["num_therm_actions"] = 0

        network = sac_network_map['leaky']
        agent = SACAgent(self.agent_params, network, chkpt_dir='test/agent_tests/test_results')

        ep_model = self.setup_env('test/eplus_files/test_control.idf')
        ep_model.set_blinds(
            ["WF-1"],
            blind_material_name="White Painted Metal Blind",
            agent_control=True
        )

        observations, actions, agent = self.run_episode(ep_model, agent, 'SAT_SP')
        obs_test = pd.DataFrame.from_dict(observations)

        sat_actions, therm_actions, blind_actions = actions

        obs_test['actions'] = [a1 for a1, _ in sat_actions]
        obs_test['sat_stpts'] = [a2.item() for _, a2 in sat_actions]
        obs_test['blind_actions'] = [a1[0] for a1 in blind_actions]

        # obs_test.to_csv('test/agent_tests/test_results/sac_blinds_obs.csv', index=False)

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

        obs_true = pd.read_csv('test/agent_tests/saved_results/sac_blinds_obs.csv')
        for c in float_cols:
            close = np.isclose(obs_test[c], obs_true[c])
            self.assertTrue(close.all())

    def test_sac_blinds_many_zone_single_setpoint(self):
        self.agent_params["num_sat_actions"] = 1
        self.agent_params["num_blind_actions"] = 1
        self.agent_params["num_therm_actions"] = 0

        network = sac_network_map['leaky']
        agent = SACAgent(self.agent_params, network, chkpt_dir='test/agent_tests/test_results')

        ep_model = self.setup_env('test/eplus_files/5Zone_Control_SAT_no_windowcontrol.idf')
        ep_model.set_blinds(
            ["WF-1", "WB-1", "WL-1", "WR-1"],
            blind_material_name="White Painted Metal Blind",
            agent_control=True
        )
        # pdb.set_trace()

        observations, actions, agent = self.run_episode(ep_model, agent, 'SAT_SP', blinds_zone_multi=True)
        obs_test = pd.DataFrame.from_dict(observations)

        sat_actions, therm_actions, blind_actions = actions

        obs_test['actions'] = [a1 for a1, _ in sat_actions]
        obs_test['sat_stpts'] = [a2.item() for _, a2 in sat_actions]
        obs_test['blind_actions'] = [a1[0] for a1 in blind_actions]

        self.assertEqual(len(blind_actions[0]), 1)

        self.assertTrue(obs_test['Blind Angle Zone 1'].equals(obs_test['Blind Angle Zone 2']))
        self.assertTrue(obs_test['Blind Angle Zone 1'].equals(obs_test['Blind Angle Zone 3']))
        self.assertTrue(obs_test['Blind Angle Zone 1'].equals(obs_test['Blind Angle Zone 4']))

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

        obs_true = pd.read_csv('test/agent_tests/saved_results/sac_blinds_multi_zone_single_stpt_obs.csv')
        for c in float_cols:
            close = np.isclose(obs_test[c], obs_true[c])
            self.assertTrue(close.all())

    # def test_sac_blinds_many_zone_many_setpoint(self):
    #     self.agent_params["num_sat_actions"] = 1
    #     self.agent_params["num_blind_actions"] = 4
    #     self.agent_params["num_therm_actions"] = 0
    #
    #     network = sac_network_map['leaky']
    #     agent = SACAgent(self.agent_params, network, chkpt_dir='test/agent_tests/test_results')
    #     ep_model = self.setup_env('test/eplus_files/5Zone_Control_SAT_no_windowcontrol.idf')
    #     ep_model.set_blinds(
    #         ["WF-1", "WR-1", "WB-1", "WL-1"],
    #         blind_material_name="White Painted Metal Blind",
    #         agent_control=True
    #     )
    #     # pdb.set_trace()
    #
    #     observations, actions, agent = self.run_episode(ep_model, agent, 'SAT_SP',
    #                                                     blinds_zone_multi=True, blinds_stpt_multi=True)
    #     obs_test = pd.DataFrame.from_dict(observations)
    #
    #     sat_actions, therm_actions, blind_actions = actions
    #
    #     obs_test['actions'] = [a1 for a1, _ in sat_actions]
    #     obs_test['sat_stpts'] = [a2.item() for _, a2 in sat_actions]
    #     for i in range(0, len(blind_actions[0])):
    #         obs_test[f'Blind Action {i+1}'] = [a1[i] for a1 in blind_actions]
    #
    #     self.assertEqual(len(blind_actions[0]), 4)
    #
    #     # pdb.set_trace()
    #
    #     self.assertFalse(obs_test['Blind Angle Zone 1'].equals(obs_test['Blind Angle Zone 2']))
    #     self.assertFalse(obs_test['Blind Angle Zone 1'].equals(obs_test['Blind Angle Zone 3']))
    #     self.assertFalse(obs_test['Blind Angle Zone 1'].equals(obs_test['Blind Angle Zone 4']))
    #
    #     float_cols = [
    #         'Outdoor Temp.',
    #         'Diff. Solar Rad.',
    #         'Direct Solar Rad.',
    #         'Indoor Temp.',
    #         'Indoor Temp. Setpoint',
    #         'PPD',
    #         'Occupancy Flag',
    #         'Blind Angle Zone 1',
    #         'Heat Coil Power',
    #         'HVAC Power',
    #         'Sys Out Temp.',
    #         'MA Temp.',
    #         'actions',
    #         'sat_stpts',
    #         'Blind Action 1',
    #         'Blind Action 2',
    #         'Blind Action 3',
    #         'Blind Action 4'
    #     ]
    #
    #     obs_true = pd.read_csv('test/agent_tests/saved_results/sac_blinds_multi_zone_multi_stpt_obs.csv')
    #     for c in float_cols:
    #         print(c)
    #         close = np.isclose(obs_test[c], obs_true[c])
    #         self.assertTrue(close.all())

    # def test_sac_thermostat(self):
    #     self.agent_params["num_sat_actions"] = 0
    #     self.agent_params["num_blind_actions"] = 0
    #     self.agent_params["num_therm_actions"] = 1
    #
    #     network = sac_network_map['leaky']
    #     agent = SACAgent(self.agent_params, network, chkpt_dir='test/agent_tests/test_results')
    #
    #     ep_model = self.setup_env()

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

    @staticmethod
    def run_episode(ep_model, agent, control_type,
                    blinds_zone_multi=False, blinds_stpt_multi=False, therm_stpt_multi=False):

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
            if control_type == 'SAT_SP':
                stpt_actions = SatAction(sat_actions[0][1], obs, sat_actions[0][0])
            else:
                therm_actions = ThermActionMultiZone(therm_actions, obs)
            blind_actions = BlindAction(blind_actions, obs)

            env_actions += stpt_actions
            env_actions += therm_actions
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

    @staticmethod
    def create_thermostat_actions(season):
        actions = []
        for i in range(1, 6):
            actions.append({
                "priority": 0,
                "component_type": "Zone Temperature Control",
                "control_type": f"{season.capitalize()} Setpoint",
                "actuator_key": f"SPACE{i}-1"
            })
        return actions


if __name__ == '__main__':
    unittest.main()
