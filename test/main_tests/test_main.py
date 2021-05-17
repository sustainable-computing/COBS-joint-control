import unittest, pdb, os
import pandas as pd
import numpy as np

from main import setup, run_continuous
from test.test_config import state_name, forecast_vars, eplus_var_types, eplus_naming_dict


class TestMain(unittest.TestCase):

    planning_steps = 10
    control_types = ['SAT_SP']
    agent_cases = ['PPO', 'DuelingDQN', 'SAC']
    blind_cases = ['True', 'False']
    daylight_cases = ['True', 'False']
    season_cases = ['heating', 'cooling']
    default_args = [
        '--power_mult', '0.1',
        '--therm_mult', '0.2',
        '--vis_mult', '0.3',
        # '--save_root', 'test/main_tests/saved_results_multi_zone_blinds',
        '--save_root', 'test/main_tests/test_results',
        '--end_run', '1',
        '--reward_type', 'OCTO',
        '--network', 'leaky',
        '--eplus_path', '/Applications/EnergyPlus-9-3-0-bugfix/',
        # '--idf_path', 'eplus_files/5Zone_Control_Therm.idf',
        '--planning_steps', str(planning_steps)
    ]

    def test_setup_satsp(self):
        ct = 'SAT_SP'
        for a in self.agent_cases:
            for b in self.blind_cases:
                for d in ['False']:
                    for s in self.season_cases:
                        args = self.default_args + ['--agent_type', a]
                        args = args + ['--daylighting', d]
                        args = args + ['--season', s]
                        args = args + ['--blinds', b]
                        args = args + ['--control_type', ct]
                        args = args + ['--zone_blinds_multi', 'False']
                        n = self.set_network(a)
                        args = args + ['--network', n]

                        setup_result = setup(args)

                        self._check_model(setup_result[0], b, d, s, 1, ct)
                        self._check_agent(setup_result[1], setup_result[3], a)
                        self._check_forecast(setup_result[2])
                        self.assertEqual(ct, setup_result[4])
                        self._check_other_args(setup_result[5], a, n, s, b, d, ct)
                        # base = f"test/main_tests/test_results/rl_results/{a}_{n}_{s}_blinds{b}_dlighting{d}_0.1_0.2_0.3"
                        base = f"test/main_tests/test_results/rl_results/{ct}_{a}_{n}_{s}_" \
                               f"blinds{b}MultiFalse_dlighting{d}_0.1_0.2_0.3"
                        self.assertEqual(setup_result[5][2], base)

    def test_setup_thermsp(self):
        ct = 'THERM_SP'
        for a in self.agent_cases:
            for b in self.blind_cases:
                for d in ['False']:
                    for s in self.season_cases:
                        for thrm_multi in ['True', 'False']:
                            for blind_multi in ['True', 'False']:
                                args = self.default_args + ['--agent_type', a]
                                args = args + ['--daylighting', d]
                                args = args + ['--season', s]
                                args = args + ['--blinds', b]
                                args = args + ['--control_type', ct]
                                args = args + ['--control_therm_multi', thrm_multi]
                                args = args + ['--control_blinds_multi', blind_multi]
                                args = args + ['--zone_blinds_multi', 'False']
                                n = self.set_network(a)
                                args = args + ['--network', n]

                                print('========', a, b, d, s, thrm_multi, blind_multi)
                                setup_result = setup(args)

                                self._check_model(setup_result[0], b, d, s, 4, ct)
                                self._check_agent(setup_result[1], setup_result[3], a)
                                self._check_forecast(setup_result[2])
                                self.assertEqual(ct, setup_result[4])
                                self._check_other_args(setup_result[5], a, n, s, b, d, ct)
                                base = f"test/main_tests/test_results/rl_results/{ct}Multi{thrm_multi}_{a}_{n}_{s}_" \
                                       f"blinds{b}Multi{blind_multi}_dlighting{d}_0.1_0.2_0.3"
                                self.assertEqual(setup_result[5][2], base)

    def test_run_sac(self):
        # ct = 'SAT_SP'
        for ct in ['SAT_SP']:
            for a in ['SAC']:
                for b in self.blind_cases:
                    for d in ['False']:
                        for s in self.season_cases:
                            # for bm in ['True', 'False']:
                            args = self.default_args + ['--testing', 'True']
                            args = args + ['--agent_type', a]
                            args = args + ['--daylighting', d]
                            args = args + ['--season', s]
                            args = args + ['--blinds', b]
                            args = args + ['--control_type', ct]
                            args = args + ['--zone_blinds_multi', 'False']
                            # args = args + ['--control_blinds_multi', bm]
                            n = self.set_network(a)
                            args = args + ['--network', n]

                            print('=======', b, d, s)
                            ep_model, agent, forecast_state, agent_type, control, args = setup(args)
                            ep_model.edit_configuration('SCHEDULE:COMPACT', {'Name': 'ReheatCoilAvailSched'}, {
                                'Field 4': 1
                            })
                            run_continuous(ep_model, agent, forecast_state, control, args)
                            self._check_run_agent(args)

    def test_run_dueling_dqn_multi_blinds(self):
        # ct = 'SAT_SP'
        for ct in ['SAT_SP']:
            for a in ['DuelingDQN']:
                for b in self.blind_cases:
                    for d in ['False']:
                        for s in self.season_cases:
                            # for bm in ['True', 'False']:
                            args = self.default_args + ['--testing', 'True']
                            args = args + ['--agent_type', a]
                            args = args + ['--daylighting', d]
                            args = args + ['--season', s]
                            args = args + ['--blinds', b]
                            args = args + ['--control_type', ct]
                            args = args + ['--zone_blinds_multi', 'False']
                            # args = args + ['--control_blinds_multi', bm]
                            n = self.set_network(a)
                            args = args + ['--network', n]

                            print('=======', b, d, s)
                            ep_model, agent, forecast_state, agent_type, control, args = setup(args)
                            ep_model.edit_configuration('SCHEDULE:COMPACT', {'Name': 'ReheatCoilAvailSched'}, {
                                'Field 4': 1
                            })
                            run_continuous(ep_model, agent, forecast_state, control, args)
                            # self._check_run_agent(args)

    def test_run_sac_multi_blinds(self):
        # ct = 'SAT_SP'
        for ct in ['SAT_SP']:
            for a in ['SAC']:
                for b in self.blind_cases:
                    for d in ['False']:
                        for s in self.season_cases:
                            for bm in ['True', 'False']:
                                args = self.default_args + ['--testing', 'True']
                                args = args + ['--agent_type', a]
                                args = args + ['--daylighting', d]
                                args = args + ['--season', s]
                                args = args + ['--blinds', b]
                                args = args + ['--control_type', ct]
                                # args = args + ['--zone_blinds_multi', 'False']
                                args = args + ['--control_blinds_multi', bm]
                                # "--automatic_entropy_tuning", 'False',
                                args = args + ["--automatic_entropy_tuning", 'False']
                                n = self.set_network(a)
                                args = args + ['--network', n]

                                print('=======', b, d, s)
                                ep_model, agent, forecast_state, agent_type, control, args = setup(args)
                                ep_model.edit_configuration('SCHEDULE:COMPACT', {'Name': 'ReheatCoilAvailSched'}, {
                                    'Field 4': 1
                                })
                                run_continuous(ep_model, agent, forecast_state, control, args)
                                # self._check_run_agent(args)
    #
    # def test_run_ppo(self):
    #     ct = 'SAT_SP'
    #     for a in ['PPO']:
    #         for b in self.blind_cases:
    #             for d in ['False']:
    #                 for s in self.season_cases:
    #                     args = self.default_args + ['--testing', 'True']
    #                     args = args + ['--agent_type', a]
    #                     args = args + ['--daylighting', d]
    #                     args = args + ['--season', s]
    #                     args = args + ['--blinds', b]
    #                     args = args + ['--control_type', ct]
    #                     args = args + ['--zone_blinds_multi', 'False']
    #                     n = self.set_network(a)
    #                     args = args + ['--network', n]
    #
    #                     ep_model, agent, forecast_state, agent_type, _, args = setup(args)
    #                     ep_model.edit_configuration('SCHEDULE:COMPACT', {'Name': 'ReheatCoilAvailSched'}, {
    #                         'Field 4': 1
    #                     })
    #                     run_continuous_old(ep_model, agent, forecast_state, args)
    #                     self._check_run_agent(args)

    # def test_run_dqn(self):
    #     ct = 'SAT_SP'
    #     for a in ['DuelingDQN']:
    #         for b in self.blind_cases:
    #             for d in ['False']:
    #                 for s in self.season_cases:
    #                     args = self.default_args + ['--testing', 'True']
    #                     args = args + ['--agent_type', a]
    #                     args = args + ['--daylighting', d]
    #                     args = args + ['--season', s]
    #                     args = args + ['--blinds', b]
    #                     args = args + ['--control_type', ct]
    #                     args = args + ['--zone_blinds_multi', 'False']
    #                     n = self.set_network(a)
    #                     args = args + ['--network', n]
    #
    #                     ep_model, agent, forecast_state, agent_type, _, args = setup(args)
    #                     ep_model.edit_configuration('SCHEDULE:COMPACT', {'Name': 'ReheatCoilAvailSched'}, {
    #                         'Field 4': 1
    #                     })
    #                     run_continuous_old(ep_model, agent, forecast_state, args)
    #                     self._check_run_agent(args)

    def set_network(self, a):
        if a == 'PPO':
            n = 'sequential'
        if a == 'SAC':
            n = 'leaky'
        if a == 'DuelingDQN':
            n = 'octo'
        return n

    def _check_model(self, model, b, d, s, winlen, ct):
        # Check blinds
        if b == 'True':
            shading_control_type = 'OnIfScheduleAllows'
            shading_is_scheduled = 'YES'
        else:
            shading_control_type = 'AlwaysOff'
            shading_is_scheduled = 'NO'
        blind_obj = model.get_configuration('WINDOWSHADINGCONTROL')[0]
        self.assertEqual(len(model.get_configuration('WINDOWSHADINGCONTROL')), winlen)
        self.assertEqual(blind_obj['Shading_Control_Type'], shading_control_type)
        self.assertEqual(blind_obj['Shading_Control_Is_Scheduled'].upper(), shading_is_scheduled)

        # Check daylighting
        if d == 'True':
            dlight = 1
        else:
            dlight = 0
        self.assertEqual(int(model.get_configuration('SCHEDULE:COMPACT', 'DaylightingAvail')['Field_4']), dlight)

        # Check season
        if s == 'heating':
            heat = 1
            cool = 0
            run_period = (32, 1991, 1, 1)
        else:
            heat = 0
            cool = 1
            run_period = (32, 1991, 7, 1)
        run_obj = model.get_configuration('RunPeriod')[0]
        self.assertEqual(run_obj['Begin_Month'], run_period[2])
        self.assertEqual(run_obj['End_Month'], run_period[2] + 1)
        self.assertEqual(run_obj['Begin_Day_of_Month'], run_period[3])

        if ct == 'SAT_SP':
            self.assertEqual(int(model.get_configuration('SCHEDULE:COMPACT', 'HeatingCoilAvailSched')['Field_4']), heat)
            self.assertEqual(int(model.get_configuration('SCHEDULE:COMPACT', 'CoolingCoilAvailSched')['Field_4']), cool)

        else:
            self.assertEqual(model.get_configuration('SCHEDULE:COMPACT', 'HeatingCoilAvailSched')['Field_4'], '1.0')
            self.assertEqual(model.get_configuration('SCHEDULE:COMPACT', 'CoolingCoilAvailSched')['Field_4'], '1.0')

    def _check_agent(self, agent, agent_type, a):
        self.assertEqual(a, agent_type)

        # TODO will probably check these by running each case individually
        if a == 'SAC':
            pass
        elif a == 'PPO':
            pass
        elif a == 'DuelingDQN':
            pass

    def _check_forecast(self, forecast):
        self.assertEqual(len(forecast), len(forecast_vars) * self.planning_steps)

    def _check_other_args(self, args, a, n, s, b, d, ct):
        start_run, end_run, base_name, blinds, _ = args
        self.assertEqual(start_run, 0)
        self.assertEqual(end_run, 1)
        self.assertEqual(str(blinds), b)

    def _check_run_agent(self, args, multi_zone_blinds=False):
        test_result_pth = args[2]
        base_name = os.path.basename(os.path.normpath(test_result_pth))
        if multi_zone_blinds:
            true_result_pth = os.path.join('test', 'main_tests', 'saved_results_multi_zone_blinds',
                                           'rl_results', base_name)
        else:
            true_result_pth = os.path.join('test', 'main_tests', 'saved_results', 'rl_results', base_name)
        # true_result_pth = os.path.join('test', 'main_tests', 'saved_results', 'rl_results', base_name[7:])

        test_result = pd.read_csv(os.path.join(test_result_pth, 'run_0.csv'), index_col='Unnamed: 0')
        true_result = pd.read_csv(os.path.join(true_result_pth, 'run_0.csv'), index_col='Unnamed: 0')

        # pdb.set_trace()

        for col in true_result.columns:
            if col == 'time':
                # skip reward because it has a NaN value that cannot be compare with isclose
                # time cannot be compare because it is an object
                continue
            close = np.isclose(test_result[col], true_result[col], equal_nan=True, rtol=1e-3)
            try:
                self.assertTrue(close.all())
            except AssertionError:
                pdb.set_trace()
