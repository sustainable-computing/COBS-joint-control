import os, argparse, sys, json, shutil, pdb
import numpy as np
import pandas as pd
import torch
import glob

from cobs.model import Model
from cobs.occupancy_generator import OccupancyGenerator as OG
from cobs.predictive_model.csv_importer import CsvImporter
from default_config import state_name, forecast_vars, eplus_naming_dict, eplus_var_types, \
    all_agent_params, agent_map, reward_map, stpt_action, blind_action, dqn_network_map, dqn_blinds_network_map, \
    sac_network_map, ppo_network_map, branching_dueling_dqn_network_map, \
    SatAction, BlindAction, ThermCoolAction, ThermHeatAction, blind_object_list, blind_schedules, zones, VAVAction

from test.test_config import eplus_var_types as eplus_var_types_test
from test.test_config import eplus_naming_dict as eplus_naming_dict_test
from test.test_config import BlindActionSingleZone as BlindAction_Test
from test.test_config import ThermActionSingleZone as ThermAction_Test
from test.test_config import BlindActionMultiZone


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def obs_to_state_values(obs, target_states, print_out=False):
    state_values = list()
    for name in target_states:
        if isinstance(name, str) and name in obs:
            if name == "time":
                state_values.append(obs[name].hour)
                if print_out: print(f"{name}: {obs[name].hour}")
            else:
                state_values.append(obs[name])
                if print_out: print(f"{name}: {obs[name]}")
        elif isinstance(name, dict):
            for category, value in name.items():
                if category not in obs:
                    continue
                for sub_name in value:
                    if sub_name not in obs[category]:
                        continue
                    state_values.append(obs[category][sub_name])
                    if print_out: print(f"{category}-{sub_name}: {obs[category][sub_name]}")
    return state_values


def setup(args):
    parser = argparse.ArgumentParser(description='RL EPlus Environment Params')
    # RUN PARAMS
    parser.add_argument('--start_run', type=int, default=0,
                        help='Number run to start on. Defaults to 0. Can use this to continue training.')
    parser.add_argument('--end_run', type=int, required=True,
                        help='Number run to end on. (Note: if used with start_run > 0,'
                             ' this is not the total number of runs, it is the number of the final run.)')
    # parser.add_argument('--base_name', type=str, required=True,
    #                     help='This will be prepended to the directory name to specify the run type.'
    #                          'The rest of the directory name contains hyperparameter info. ')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default 42)')
    parser.add_argument('--chkpt_dir', type=str, default='checkpoints',
                        help='Directory for loading agent checkpoints (default checkpoints)')

    # AGENT PARAMS
    parser.add_argument('--agent_type', type=str, required=True,
                        help='SAC | DQN | DDQN | DuelingDQN | PPO')
    parser.add_argument('--alpha', type=float, default=0.2,
                        help='SAC Param: Temperature parameter alpha determines the relative importance of the entropy '
                             'term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=str, default='True', metavar='G',
                        help='SAC Param: Automatically adjust alpha (default: True)')
    parser.add_argument('--min_action', type=int, default=-20,
                        help='The lowest valued action term against the reward (default: 20)')
    parser.add_argument('--max_action', type=int, default=20,
                        help='The highest valued action term against the reward (default: 65)')
    parser.add_argument('--network', type=str, required=True,
                        help='no_relu | leaky | sequential')

    # REWARD PARAMS
    parser.add_argument('--reward_type', default='Coil',
                        help='Reward type: Coil | Action | PPD | OCTO (default: Coil)')
    parser.add_argument('--reward_param', type=float, default=0.5,
                        help='Relative weight parameter for discomfort vs energy use.'
                             'Different magnitudes for different reward functions are expected.'
                             'Double check the reward objects to see the magnitudes')
    parser.add_argument('--power_mult', type=float, help='Required for OCTO reward')
    parser.add_argument('--therm_mult', type=float, help='Required for OCTO reward')
    parser.add_argument('--vis_mult', type=float, help='Required for OCTO reward')

    # ENVIRONMENT PARAMS
    parser.add_argument('--planning_steps', type=int, default=12,
                        help='The amount of look ahead time steps (default 12)')
    parser.add_argument('--save_root', type=str, default='',
                        help='The directory to save results and models.')

    # MODEL PARAMS
    parser.add_argument('--eplus_path', type=str, default='/Applications/EnergyPlus-9-3-0-bugfix/',
                        help='The install location of EnergyPlus (default /Applications/EnergyPlus-9-3-0-bugfix/)')
    # parser.add_argument('--idf_path', type=str, default='eplus_files/5Zone_Control.idf',
    #                     help='The path to the idf to run')
    parser.add_argument('--epw_path', type=str, default='eplus_files/pittsburg_TMY3.epw',
                        help='The path to the epw to run')

    # CASE PARAMS
    # parser.add_argument('--control_type', type=str, required=True, help='SAT_SP | THERM_SP '
    #                                                                     'SAT_SP: Controls the supply air temperature setpoint'
    #                                                                     'THERM_SP: Controls the thermostat setpoints')
    parser.add_argument('--control_sat', type=str, default='True',
                        help='Set to False if the agent do not control the SAT_SP')
    parser.add_argument('--control_therm', type=str, default='True',
                        help='Set to False if the agent do not control the THERM_SP')
    parser.add_argument('--season', type=str, required=True, help='heating | cooling')
    parser.add_argument('--blinds', type=str, required=True, help='True | False')
    parser.add_argument('--control_blinds_multi', type=str, default='False', help='True | False')
    parser.add_argument('--zone_blinds_multi', type=str, default='True',
                        help='True | False')  # TODO false option does not work right now
    parser.add_argument('--daylighting', type=str, required=True, help='True | False')

    parser.add_argument('--testing', type=str, default='False', help='Set to True if running agent tests')
    parser.add_argument('--random_occupancy', type=str, default='False', help='Set to True if using random occupancy')
    parser.add_argument('--multi_agent', type=str, default='False', help='Set to True if multi-agent')
    parser.add_argument('--load_sat', type=str, default='False', help='Set to False if train the SAT_SP together')
    parser.add_argument('--load_sat_path', type=str, default='./scratch/tzhang6/hvac_control/checkpoints',
                        help='Path of the pre-trained SAT_SP controller')

    parser.add_argument('--vav', type=str, default='False', help='Replace Thermostat to VAV position')

    args = parser.parse_args(args)
    TESTING = str2bool(args.testing)

    chkpt_dir = args.chkpt_dir
    seed = args.seed
    agent_type = args.agent_type
    eplus_path = args.eplus_path
    alpha = args.alpha
    automatic_entropy_tuning = str2bool(args.automatic_entropy_tuning)
    min_action = args.min_action
    max_action = args.max_action
    network_type = args.network
    reward_type = args.reward_type
    reward_param = args.reward_param
    start_run = args.start_run
    end_run = args.end_run
    planning_steps = args.planning_steps
    # idf_path = args.idf_path
    epw_path = args.epw_path
    # control_type = args.control_type
    season = args.season
    blinds = str2bool(args.blinds)
    control_blinds_multi = str2bool(args.control_blinds_multi)
    zone_blinds_multi = str2bool(args.zone_blinds_multi)
    multi_agent = str2bool(args.multi_agent)
    daylighting = str2bool(args.daylighting)
    save_root = args.save_root
    customize_occupancy = str2bool(args.random_occupancy)
    load_sat = str2bool(args.load_sat)
    control_sat = str2bool(args.control_sat)
    control_therm = str2bool(args.control_therm)
    load_sat_path = args.load_sat_path
    vav = str2bool(args.vav)

    RL_RESULTS_DIR = os.path.join(save_root, 'rl_results')
    CHCKPT_DIR = os.path.join(save_root, chkpt_dir)

    # =============================
    # LOAD IDF
    # =============================
    base_name = f'SAT{control_sat}_THERM{control_therm}_customOcc{customize_occupancy}_{agent_type}' \
                f'_{network_type}_{season}_blinds{blinds}Multi{control_blinds_multi}' \
                f'_dlighting{daylighting}_multiAgent{multi_agent}_seed{seed}'
    if vav:
        base_name += "_vav"
    idf_path = ''

    load_sat_path = os.path.join(load_sat_path, f'SATTrue_THERMFalse_customOcc{customize_occupancy}_{agent_type}_'
                                                f'{network_type}_{season}_blindsFalseMultiFalse_dlighting{daylighting}_'
                                                f'multiAgentFalse_{args.power_mult}_{args.therm_mult}_{args.vis_mult}'
                                                f'/Main_SAT')

    if control_sat:
        if load_sat and not os.path.isdir(load_sat_path):
            raise FileNotFoundError("Cannot find the trained SAT_SP agent")
        idf_path = 'eplus_files/5Zone_Control_SAT_no_windowcontrol.idf'

    if control_therm:
        idf_path = 'eplus_files/5Zone_Temp_Multi_no_windowcontrol_update.idf'

    if not control_sat and not control_therm:
        raise LookupError("You cannot disable both controls")

    # if control_type == 'SAT_SP':
    #     base_name = f'{control_type}_customOcc{customize_occupancy}_{agent_type}_{network_type}_{season}_' \
    #                 f'blinds{blinds}Multi{control_blinds_multi}_dlighting{daylighting}'
    #     if TESTING:
    #         if zone_blinds_multi:
    #             idf_path = 'test/eplus_files/5Zone_Control_SAT_no_windowcontrol.idf'
    #         else:
    #             idf_path = 'test/eplus_files/5Zone_Control_SAT.idf'
    #     else:
    #         if zone_blinds_multi:
    #             idf_path = 'eplus_files/5Zone_Control_SAT_no_windowcontrol.idf'
    #         else:
    #             idf_path = 'eplus_files/5Zone_Control_SAT.idf'
    # elif control_type == 'THERM_SP':
    #     base_name = f'{control_type}_Multi{control_therm_multi}_customOcc{customize_occupancy}_' \
    #                 f'{agent_type}_{network_type}_{season}_' \
    #                 f'blinds{blinds}Multi{control_blinds_multi}_dlighting{daylighting}_multiAgent{multi_agent}'
    #     idf_path = 'eplus_files/5Zone_Temp_Multi_no_windowcontrol_update.idf'
    # else:
    #     raise ValueError(f"{control_type} is not a valid control type.")

    # =============================
    # LOAD FORECASTED STATE
    # =============================
    # Note that the only variable included in the forecasted state is the season
    forecasted_path = os.path.join(
        'baselines',
        f'SAT_SP_{season}_blindsNone_setpoint0_daylightingFalse.csv'
    )

    # =============================
    # SETUP ACTIONS IN TESTING CASE
    # =============================
    if TESTING:
        if blinds:
            if zone_blinds_multi:
                BlindAction_Test.set_actuators(blind_schedules)
        # if control_therm_multi:
        #     ThermAction_Test.set_actuators(zones)

    # =============================
    # SETUP REWARD
    # =============================
    reward_params = {}
    if (reward_type == 'Action') or (reward_type == 'PPD') or (reward_type == 'Coil'):
        reward_params['occ_weight'] = reward_param
        base_name = base_name + f'_{reward_param}'
    elif reward_type == 'OCTO':
        MAX_LIGHT_POWER = 1553126.4
        if season == 'heating':
            MAX_HVAC_POWER = 24883070.64
            MAX_ILLUM = 8503.35
        else:
            MAX_HVAC_POWER = 10081693.42
            MAX_ILLUM = 7165.61
        # df = pd.read_csv(forecasted_path)
        # p_min = df['Heat Coil Power'].min() + df['Cool Coil Power'].min() + df['Lights Zone 1'].min()
        # p_min = df['Heat Coil Power'].min() + df['Cool Coil Power'].min() + df['Lights Zone 1'].min()
        # df['Illum'] = (df['Illum 1 Zone 1'] + df['Illum 1 Zone 1'] + df['Illum 1 Zone 1'] + df['Illum 1 Zone 1']) / 4

        p_max = MAX_HVAC_POWER + MAX_LIGHT_POWER
        reward_params['power_range'] = [0, p_max]
        reward_params['therm_range'] = [-3, 3]
        reward_params['vis_range'] = [0, MAX_ILLUM]

        reward_params['power_mult'] = args.power_mult
        reward_params['therm_mult'] = args.therm_mult
        reward_params['vis_mult'] = args.vis_mult
        reward_params['multi_agent'] = multi_agent
        base_name = base_name + f'_{args.power_mult}_{args.therm_mult}_{args.vis_mult}'
    else:
        raise ValueError(f'{reward_type} is not an acceptable reward_type')

    reward = reward_map[reward_type](**reward_params)

    # =============================
    # SET SEED
    # =============================
    torch.manual_seed(seed)
    np.random.seed(0)

    # =============================
    # SETUP BLINDS, DAYLIGHTING AND SEASON CASES
    # =============================
    if season == 'heating':
        reheat = 1
        heat = 1
        cool = 0
        stpt = 15
        run_period = (32, 1991, 1, 1)
        if TESTING:
            print('TESTING RUN PERIOD')
            run_period = (1, 1991, 1, 1)
    elif season == 'cooling':
        reheat = 0
        heat = 0
        cool = 1
        stpt = 50
        run_period = (32, 1991, 7, 1)
        if TESTING:
            print('TESTING RUN PERIOD')
            run_period = (1, 1991, 7, 1)
    else:
        raise ValueError(f'{season} is not a valid season')
    if daylighting:
        dlight = 1
    else:
        dlight = 0

    # This blinds stuff is only here so exactly replicate the way we were running before adding multiple zone control
    # at somepoint it can be removed
    if blinds:
        state_name.append("Blind Angle Zone 1")
        if zone_blinds_multi and control_blinds_multi:
            state_name.append("Blind Angle Zone 2")
            state_name.append("Blind Angle Zone 3")
            state_name.append("Blind Angle Zone 4")
        blind_type = 'OnIfScheduleAllows'
        is_scheduled = 'YES'
    else:
        blind_type = 'AlwaysOff'
        is_scheduled = 'NO'

    print('AGENT NAME =====', agent_type)
    # =============================
    # SETUP AGENT
    # =============================
    agents = list()
    if start_run > 1:
        # If this is a continuing simulation then don't want for a certain amount of start steps before training
        # the replay_memory object will be loaded into the state and can be sampled directly
        start_steps = 0
        epsilon = 0.1
    else:
        start_steps = 5000
        if TESTING:
            start_steps = 10
        epsilon = 1

    chkpt_pth = os.path.join(CHCKPT_DIR, base_name)
    agent_params = all_agent_params[agent_type].copy()
    agent_params['seed'] = seed
    agent_params['min_sat_action'] = min_action
    agent_params['max_sat_action'] = max_action
    agent_params['min_therm_action'] = 10 if not vav else 0
    agent_params['max_therm_action'] = 40 if not vav else 100
    agent_params['min_action'] = min_action
    agent_params['max_action'] = max_action
    agent_params['start_steps'] = start_steps
    agent_params['alpha'] = alpha
    agent_params['automatic_entropy_tuning'] = automatic_entropy_tuning
    agent_params['epsilon'] = epsilon
    state_length = sum([1 if isinstance(s, str) else len(list(s.values())[0]) for s in state_name])

    if 'SAC' in agent_type:
        network = sac_network_map[network_type]
        if TESTING:
            # Make the agent smaller so that the tests run faster
            agent_params["hidden_size"] = 2
            agent_params["replay_size"] = 200
            agent_params["batch_size"] = 10
    elif 'DuelingDQN' in agent_type:
        network = branching_dueling_dqn_network_map[network_type]
    elif 'PPO' in agent_type:
        network = ppo_network_map[network_type]
    else:
        raise ValueError(f'{agent_type} is not an acceptable agent_type')

    # Set the number of actions depending on use case
    if control_therm:  # Create agents controlling per zone therm and blind
        therm_state_length = state_length
        if multi_agent:
            therm_state_length -= 9 + (blinds and zone_blinds_multi and control_blinds_multi) * 4
        # print(f"State_length: {state_length}, Forcasted_length: {len(forecast_vars) * planning_steps}")
        # print(state_name)
        therm_num_inputs = therm_state_length + len(forecast_vars) * planning_steps
        agent_params['n_state'] = therm_num_inputs
        agent_params['input_dims'] = (therm_num_inputs,)
        agent_params["num_sat_actions"] = 0
        agent_params["num_blind_actions"] = 0

        if blinds:
            agent_params["num_blind_actions"] = 1
            if control_blinds_multi and not multi_agent:
                agent_params["num_blind_actions"] = 4

        if not multi_agent:
            if control_sat:
                agent_params["num_sat_actions"] = 1
            agent_params["num_therm_actions"] = 5
            agents.append(agent_map[agent_type](agent_params, network, chkpt_dir=chkpt_pth))
        else:
            agent_params["num_therm_actions"] = 1
            for i in range(1, 6):
                if i != 5:
                    if 'DuelingDQN' in agent_type:
                        agent_params['input_dims'] = (therm_num_inputs + 1,)
                    else:
                        agent_params['n_state'] = therm_num_inputs + 1
                else:
                    agent_params["num_blind_actions"] = 0
                    if 'DuelingDQN' in agent_type:
                        agent_params['input_dims'] = (therm_num_inputs,)
                    else:
                        agent_params['n_state'] = therm_num_inputs
                # print(f"Zone{i} {agent_params['n_state']}")
                agents.append(agent_map[agent_type](agent_params, network,
                                                    chkpt_dir=os.path.join(chkpt_pth, f"Zone{i}")))

    if control_sat and multi_agent or control_sat and not control_therm:
        sat_num_inputs = state_length + len(forecast_vars) * planning_steps
        blind_count = 0
        for check_name in state_name:
            if "Blind" in check_name:
                blind_count += 1
        if multi_agent:
            sat_num_inputs -= blind_count
            blind_count = 0
        agent_params['n_state'] = sat_num_inputs
        agent_params['input_dims'] = (sat_num_inputs,)
        agent_params["num_sat_actions"] = 1
        agent_params["num_blind_actions"] = blind_count
        agent_params["num_therm_actions"] = 0
        if load_sat:
            sat_agent = agent_map[agent_type](agent_params, network, chkpt_dir=load_sat_path)
            chkpt_max_iter = 0
            for chkpt_name in glob.glob(os.path.join(load_sat_path, '*')):
                chkpt_iter = int(chkpt_name.split('_')[-1])
                chkpt_max_iter = max(chkpt_iter, chkpt_max_iter)
            if chkpt_max_iter != 400:
                exit(1)
            sat_agent.load(chkpt_max_iter)
        else:
            sat_agent = agent_map[agent_type](agent_params, network, chkpt_dir=os.path.join(chkpt_pth, f"Main_SAT"))
            if not os.path.exists(os.path.join(chkpt_pth, f"Main_SAT")):
                os.makedirs(os.path.join(chkpt_pth, f"Main_SAT"))
        agents.append(sat_agent)
    print(len(agents))
    # Set agent specific params
    # if 'SAC' in agent_type:
    #     agent_params['start_steps'] = start_steps
    #     agent_params['alpha'] = alpha
    #     agent_params['automatic_entropy_tuning'] = automatic_entropy_tuning
    #     agent_params['n_state'] = num_inputs
    #     network = sac_network_map[network_type]
    #     if TESTING:
    #         # Make the agent smaller so that the tests run faster
    #         agent_params["hidden_size"] = 2
    #         agent_params["replay_size"] = 200
    #         agent_params["batch_size"] = 10
    # elif 'DuelingDQN' in agent_type:
    #     # Remove DQN and DDQN for now because they are not being used
    #     agent_params['epsilon'] = epsilon
    #     agent_params['input_dims'] = (num_inputs,)
    #     network = branching_dueling_dqn_network_map[network_type]
    # elif 'PPO' in agent_type:
    #     # Set the number of actions depending on use case
    #     network = ppo_network_map[network_type]
    #     agent_params['n_state'] = num_inputs
    # else:
    #     raise ValueError(f'{agent_type} is not an acceptable agent_type')
    #
    # if multi_agent:
    #     agent = []
    #     for i in range(1, 6):
    #         if i != 5:
    #             if 'DuelingDQN' in agent_type:
    #                 agent_params['input_dims'] = (num_inputs + 1,)
    #             else:
    #                 agent_params['n_state'] = num_inputs + 1
    #         else:
    #             agent_params["num_blind_actions"] = 0
    #             if 'DuelingDQN' in agent_type:
    #                 agent_params['input_dims'] = (num_inputs,)
    #             else:
    #                 agent_params['n_state'] = num_inputs
    #         # print(f"Zone{i} {agent_params['n_state']}")
    #         agent.append(agent_map[agent_type](agent_params, network, chkpt_dir=os.path.join(chkpt_dir, f"Zone{i}")))
    # else:
    #     # print(f"{agent_params['n_state']}")
    #     agent = agent_map[agent_type](agent_params, network, chkpt_dir=chkpt_pth)

    # =============================
    # SETUP MODEL
    # =============================
    Model.set_energyplus_folder(eplus_path)

    if TESTING:
        ep_model = Model(
            idf_file_name=idf_path,
            weather_file=epw_path,
            eplus_naming_dict=eplus_naming_dict_test,
            eplus_var_types=eplus_var_types_test,
            reward=reward,
            tmp_idf_path=os.path.join(RL_RESULTS_DIR, base_name)
        )
    else:
        ep_model = Model(
            idf_file_name=idf_path,
            weather_file=epw_path,
            eplus_naming_dict=eplus_naming_dict,
            eplus_var_types=eplus_var_types,
            reward=reward,
            tmp_idf_path=os.path.join(RL_RESULTS_DIR, base_name)
        )
    ep_model.set_runperiod(*run_period)

    ep_model.edit_configuration('SCHEDULE:COMPACT', {'Name': 'DaylightingAvail'}, {
        'Field 4': dlight
    })
    ep_model.edit_configuration('SCHEDULE:COMPACT', {'Name': 'ReheatCoilAvailSched'}, {
        'Field 4': reheat
    })

    if control_sat:
        ep_model.edit_configuration('SCHEDULE:COMPACT', {'Name': 'HeatingCoilAvailSched'}, {
            'Field 4': heat
        })
        ep_model.edit_configuration('SCHEDULE:COMPACT', {'Name': 'CoolingCoilAvailSched'}, {
            'Field 4': cool
        })

    if zone_blinds_multi:
        if blinds:
            ep_model.set_blinds(
                blind_object_list,
                blind_material_name="White Painted Metal Blind",
                agent_control=True
            )
        else:
            ep_model.set_blinds(
                blind_object_list,
                blind_material_name="White Painted Metal Blind",
                agent_control=False
            )
    else:
        ep_model.edit_configuration('WINDOWSHADINGCONTROL', {'Name': 'CONTROL SHADE'}, {
            'Shading Control Type': blind_type,
            'Setpoint': stpt,
            'Shading Control Is Scheduled': is_scheduled
        })

    if vav:
        ep_model.delete_configuration("ZoneControl:Thermostat")
        ep_model.delete_configuration("ThermostatSetpoint:DualSetpoint")
        for air_terminal_name in ep_model.get_available_names_under_group("AirTerminal:SingleDuct:VAV:Reheat"):
            ep_model.edit_configuration(
                idf_header_name="AirTerminal:SingleDuct:VAV:Reheat",
                identifier={"Name": air_terminal_name},
                update_values={"Zone Minimum Air Flow Input Method": "Scheduled",
                               "Minimum Air Flow Fraction Schedule Name": f"{air_terminal_name} Customized Schedule"})
            ep_model.add_configuration("Schedule:Constant",
                                       {"Name": f"{air_terminal_name} Customized Schedule",
                                        "Schedule Type Limits Name": "Fraction",
                                        "Hourly Value": 0})

    external_data = CsvImporter(forecasted_path, planstep=planning_steps)
    forecast_state = external_data.get_output_states()  # TODO - think about scaling
    ep_model.add_state_modifier(external_data)

    # =============================
    # CREATE BASE DIRECTORIES AND SAVE EXPERIMENT STATE
    # =============================
    if not os.path.exists(f'logs/{base_name}'): os.makedirs(f'logs/{base_name}')
    if not os.path.exists(CHCKPT_DIR): os.makedirs(CHCKPT_DIR)
    if not os.path.exists(RL_RESULTS_DIR): os.makedirs(RL_RESULTS_DIR)
    run_dir = os.path.join(RL_RESULTS_DIR, base_name)
    if not os.path.exists(os.path.join(run_dir)): os.makedirs(run_dir)
    if multi_agent:
        for i in range(1, 6):
            if not os.path.exists(os.path.join(chkpt_pth, f"Zone{i}")): os.makedirs(os.path.join(chkpt_pth, f"Zone{i}"))

    exp_info_pth = os.path.join(run_dir, 'experiment_info.txt')

    with open(exp_info_pth, 'w') as file:
        file.write('\n' + idf_path)
        file.write('\n' + epw_path)
        file.write('\n' + season)
        file.write('\nSAT Control Status: ' + str(control_sat))
        file.write('\nSAT Control Loaded: ' + str(load_sat))
        file.write('\nSAT Control Load Path: ' + str(load_sat_path))
        file.write('\nTHERM Control Status: ' + str(control_therm))
        file.write('\nMulti Agent: ' + str(multi_agent))
        file.write('\nMulti Blind: ' + str(control_blinds_multi))
        file.write('\nWith Blinds: ' + str(blinds))
        file.write('\nAgent Type ' + agent_type + '\n')
        file.write('\nNetwork Type ' + network_type + '\n')
        # remove forecast state from dict (PPO Only)
        write_dict = {k: agent_params[k] for k in agent_params.keys() - {'target', 'dist'}}
        file.write(json.dumps(write_dict))
        file.write('\nReward Type ' + reward_type + '\n')
        file.write(json.dumps(reward_params))

    base_name = os.path.join(RL_RESULTS_DIR, base_name)
    if customize_occupancy:
        OG(ep_model, random_seed=seed).generate_daily_schedule(add_to_model=True,
                                                               overwrite_dict={f"SPACE{i}-1": f"SPACE{i}-1 People 1"
                                                                               for i in range(1, 6)})

    ep_model.run_parameters[ep_model.run_parameters.index('-d') + 1] = os.path.join(base_name, "epresult")

    # return ep_model, agent, forecast_state, agent_type, control_type, \
    return ep_model, agents, forecast_state, agent_type, \
           (start_run, end_run, base_name, blinds, TESTING, multi_agent, season, control_sat, load_sat, vav)


def run_episodic(ep_model, agent, args):
    start_run, end_run, base_name, blinds = args

    # LOAD CHECKPOINTS
    if start_run > 1:
        agent.load(start_run - 1)

    n_step = 96  # timesteps per day
    for i in range(start_run, end_run):
        print(f'\n============\nRunning simulation number {i}...\n==============\n')
        observations = []
        actions = []

        obs = ep_model.reset()
        observations.append(obs)

        state = torch.tensor(obs_to_state_values(obs, state_name + forecast_state)).unsqueeze(0).double()
        ts = pd.to_datetime(obs["time"])
        ts = ts + pd.offsets.DateOffset(year=1991)  # TODO should not be hardcoded
        feeding_state = (state, obs, ts)

        for i_episode in range(agent.tol_eps):
            action = agent.agent_start(feeding_state, i_episode)
            for t in range(n_step):
                stpt_action['note'] = action[0]
                stpt_action['value'] = action[1]
                stpt_action['start_time'] = obs['timestep'] + 1

                env_actions = [stpt_action]
                if blinds:
                    blind_action['value'] = action[2]
                    blind_action['start_time'] = obs['timestep'] + 1
                    env_actions.append(blind_action)

                obs = ep_model.step(env_actions)
                observations.append(obs)
                state = torch.tensor(obs_to_state_values(obs, state_name + forecast_state)).unsqueeze(
                    0).double()
                ts = pd.to_datetime(obs["time"])
                ts = ts + pd.offsets.DateOffset(year=1991)  # TODO should not be hardcoded
                feeding_state = (state, obs, ts)

                if ep_model.is_terminate() or (t == (n_step - 1)):
                    agent.agent_end(obs["reward"], feeding_state, i_episode)
                else:
                    action = agent.agent_step(obs["reward"], feeding_state)
                    actions.append(action)


def save(run_dir, run_num, agents, observations, actions, TESTING, multi_agent, control_sat, load_sat):
    print('Saving...')
    # run_dir = os.path.join('rl_results', save_name)
    # if not os.path.exists(run_dir): os.makedirs(run_dir)
    if run_num % 100 == 0:
        for i in range(len(agents)):
            if i == len(agents) - 1 and control_sat and load_sat:
                continue
            agents[i].save(run_num)
    all_obs_df = pd.DataFrame(observations)
    if TESTING:
        r = ['reward']
        d = list(eplus_naming_dict_test.values()) + ['time']
    else:
        r = ['reward agent 1']
        if multi_agent:
            r = [f"reward agent {i + 1}" for i in range(5)]
        d = list(eplus_naming_dict.values()) + ['time', 'total reward'] + r

    obs_df = all_obs_df[d].copy()

    obs_df['run'] = run_num

    obs_df['time'] = obs_df['time'].mask(obs_df['time'].dt.year > 1,  # Warn: hacky way of replacing year
                                         obs_df['time'] + pd.offsets.DateOffset(year=1991))

    sat_actions, therm_actions, blind_actions = actions

    if sat_actions:
        obs_df['Action'] = [a1 for a1, _ in sat_actions]
        obs_df['SAT STPT'] = [a2.item() for _, a2 in sat_actions]
    for i in range(0, len(therm_actions[0])):
        obs_df[f'THERM STPT {i + 1}'] = [a1[i] for a1 in therm_actions]
    for i in range(0, len(blind_actions[0])):
        obs_df[f'Blind Action {i + 1}'] = [a1[i] for a1 in blind_actions]

    mode = 'a' if run_num % 100 != 0 else 'w'

    obs_df.to_csv(os.path.join(run_dir, f'run_{run_num // 100}.csv'), mode=mode, header=mode == 'w')
    with open(os.path.join(run_dir, 'convergence.csv'), mode) as conv_file:
        r_data = obs_df[r].iloc[-1].tolist()
        if len(r_data) > 1:
            r_data.append(sum(r_data))
        conv_file.write(f"{','.join(map(str, r_data))}\n")


def run_continuous(ep_model, agents, forecast_state, args):
    # print(len(agents))
    start_run, end_run, base_name, blinds, TESTING, multi_agent, season, control_sat, load_sat, vav = args
    if not isinstance(agents, list):
        agents = [agents]

    agent_state_name_list = list()
    for i, agent in enumerate(agents, start=1):
        if multi_agent and i <= 5:
            agent_state_name = [f"Occu Zone {i}", f"Temp Zone {i}", "time"]
            if i != 5: agent_state_name.append(f"Blind Angle Zone {i}")
            agent_state_name_list.append(agent_state_name)
        elif i == len(agents) and control_sat and load_sat:
            agent_state_name = []
            for name in state_name:
                if not isinstance(name, str) or "Blind" not in name:
                    agent_state_name.append(name)
            agent_state_name_list.append(agent_state_name)
        else:
            agent_state_name_list.append(state_name)

    ba = BlindAction
    ta = {"heating": ThermHeatAction,
          "cooling": ThermCoolAction} if not vav else VAVAction
    if TESTING:
        ba = BlindAction_Test
        ta = [ThermHeatAction, ThermCoolAction]  # TODO

    # control_type, control_blinds_multi, control_therm_multi = control

    # LOAD CHECKPOINTS
    if start_run > 1:
        for i in range(len(agents)):
            if i == len(agents) - 1 and control_sat and load_sat:
                continue
            agents[i].load(start_run - 1)

    if baseline:
        end_run = start_run

    for run_num in range(start_run, end_run + 1):
        print(f'\n============\nRunning simulation number {run_num}, {base_name}...\n==============\n')
        observations = []
        # actions = []
        sat_actions_list = []
        blind_actions_list = []
        therm_actions_list = []

        obs = ep_model.reset()
        # observations.append(obs)

        sat_actions = list()
        therm_actions = list()
        blind_actions = list()
        for i, agent in enumerate(agents):
            if i == len(agents) - 1 and control_sat and load_sat:
                state = torch.tensor(obs_to_state_values(obs, agent_state_name_list[i] + forecast_state)).double()
                action = agent.inference_only((state, obs, 0))
            else:
                state = torch.tensor(obs_to_state_values(obs, agent_state_name_list[i] + forecast_state)).double()
                action = agent.agent_start((state, obs, 0))

            sat_sub_actions, therm_sub_actions, blind_sub_actions = action
            if sat_sub_actions and sat_sub_actions[0] and sat_sub_actions[0][0]:
                sat_actions.extend(sat_sub_actions)
            therm_actions.extend(therm_sub_actions)
            blind_actions.extend(blind_sub_actions)

        if baseline:
            sat_actions = base_sat.copy()
            therm_actions = base_stpt.copy()
            blind_actions = base_angle.copy()

        # sat_actions, therm_actions, blind_actions = action
        # pprint.pprint(obs)

        # sat_actions_list.append(sat_actions[0])
        # therm_actions_list.append(therm_actions)
        # blind_actions_list.append(blind_actions)

        # actions.append(action)
        while not ep_model.is_terminate():
            # # SETUP ACTIONS
            env_actions = []
            if therm_actions:
                if not vav:
                    for control_season in ta:
                        if control_season == season:
                            acting_actions = therm_actions
                        else:
                            acting_actions = len(therm_actions) * [50 if control_season == "cooling" else 5]
                        env_actions += ta[control_season](acting_actions, obs)
                else:
                    for i in range(len(therm_actions)):
                        therm_actions[i] /= 100
                    env_actions += ta(therm_actions, obs)
            if blind_actions:
                blind_actions = ba(blind_actions, obs)
                env_actions += blind_actions
            if sat_actions and sat_actions[0] and sat_actions[0][0]:
                stpt_actions = SatAction(sat_actions[0][1], obs, sat_actions[0][0])
                env_actions += stpt_actions

            # print(env_actions)
            obs = ep_model.step(env_actions)

            # print(agents)
            for i in range(len(agents)):
                # print("?")
                if i == len(agents) - 1 and control_sat and load_sat:
                    continue
                obs[f"reward agent {i + 1}"] = obs["reward"][i]

                # Added for model-based model to print RMSE
                if agents[i].type == 'BDQNwPlanningAgent':
                    agents[i].get_world_rmse(obs)

            obs["total reward"] = sum(obs["reward"])

            observations.append(obs)
            # print(obs)

            sat_actions = list()
            therm_actions = list()
            blind_actions = list()
            for i, agent in enumerate(agents):
                if i == len(agents) - 1 and control_sat and load_sat:
                    state = torch.tensor(obs_to_state_values(obs, agent_state_name_list[i] + forecast_state)).double()
                    feeding_state = (state, obs, obs["timestep"])
                    action = agent.inference_only(feeding_state)
                else:
                    state = torch.tensor(obs_to_state_values(obs, agent_state_name_list[i] + forecast_state)).double()
                    feeding_state = (state, obs, obs["timestep"])
                    action = agent.agent_step(obs["reward"][i], feeding_state)

                sat_sub_actions, therm_sub_actions, blind_sub_actions = action
                if sat_sub_actions and sat_sub_actions[0] and sat_sub_actions[0][0]:
                    sat_actions.extend(sat_sub_actions)
                therm_actions.extend(therm_sub_actions)
                blind_actions.extend(blind_sub_actions)

            if baseline:
                sat_actions = base_sat.copy()
                therm_actions = base_stpt.copy()
                blind_actions = base_angle.copy()

            # sat_actions, therm_actions, blind_actions = action
            if sat_actions:
                sat_actions_list.append(sat_actions[0])
            therm_actions_list.append(therm_actions)
            blind_actions_list.append(blind_actions)

        # SAVE SAVE RESULT
        actions = (sat_actions_list, therm_actions_list, blind_actions_list)
        save(base_name, run_num, agents, observations, actions, TESTING, multi_agent, control_sat, load_sat)

    print("Done!")


if __name__ == "__main__":
    print('\n  Starting...')
    import pprint

    baseline = False
    base_sat = []
    base_stpt = [22] * 5
    base_angle = [5] * 4
    ep_model, agents, forecast_state, agent_type, args = setup(sys.argv[1:])
    run_continuous(ep_model, agents, forecast_state, args)
