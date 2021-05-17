from agents.SACAgent import SACAgent
from agents.DDQNAgent import DDQNAgent
from agents.DQNAgent import DQNAgent
from agents.DQNAgentBlinds import DQNAgentBlinds
from agents.BranchingDuelingDQNAgent import BranchingDuelingDQNAgent
from agents.PPOAgent3 import PPOAgent
from agents.PPOAgentBlinds import PPOAgentBlinds
# from agents.PPOAgent import PPOAgent

from agents.networks.dueling_dqn_networks import BranchingDuelingDeepQOcto
from agents.networks.dqn_networks import DeepQ, DeepQLeaky, DeepQSequential
from agents.networks.dqn_blinds_networks import DeepQ_Blinds, DeepQLeaky_Blinds, DeepQSequential_Blinds
from agents.networks.sac_networks import DoubleQ, DoubleQLeaky, DoubleQSequential
from agents.networks.ppo_networks import ActorNetwork, CriticNetwork

from utils.ActionCreator import ActionCreator
from utils.rewards import ViolationPActionReward, ViolationPCoilReward, PPDPActionReward, OctoReward

# name of blind objects and zones
blind_object_list = ["WF-1", "WR-1", "WB-1", "WL-1"]
blind_schedules = [f"{b}_shading_schedule" for b in blind_object_list]
zones = ["SPACE1-1", "SPACE2-1", "SPACE3-1", "SPACE4-1", "SPACE5-1"]

# Used to test actions for a single zone only
SatAction = ActionCreator("Schedule:Constant", "Schedule Value", "SAT_SP")
BlindActionSingleZone = ActionCreator("Schedule:Constant", "Schedule Value", "WF-1_shading_schedule")
ThermActionSingleZone = ActionCreator("Zone Temperature Control", "Heating Setpoint", "SPACE1-1")
BlindAction = ActionCreator("Schedule:Constant", "Schedule Value", blind_schedules)

# Used to test a single setpoint across multiple zones, or different setpoints for each zone
BlindActionMultiZone = ActionCreator("Schedule:Constant", "Schedule Value", blind_schedules)
ThermActionMultiZone = ActionCreator("Zone Temperature Control", "Heating Setpoint", zones)

state_name = [
    "Indoor Temp.",
    # "MA Temp.",
]

forecast_vars = [
    "Outdoor Temp.",
    "Total Rad.",
    "Occupancy Flag",
]

eplus_naming_dict = {
    ('Site Outdoor Air Drybulb Temperature', '*'): "Outdoor Temp.",
    ('Site Diffuse Solar Radiation Rate per Area', '*'): "Diff. Solar Rad.",
    ('Site Direct Solar Radiation Rate per Area', '*'): "Direct Solar Rad.",
    ('Facility Total HVAC Electric Demand Power', '*'): "HVAC Power",
    ('System Node Temperature', 'VAV SYS 1 OUTLET NODE'): "Sys Out Temp.",
    ('Heating Coil Electric Power', 'Main Heating Coil 1'): "Heat Coil Power",
    ('Cooling Coil Electric Power', 'Main Cooling Coil 1'): "Cool Coil Power",
    ('Occupancy Flag', '*'): "Occupancy Flag",
    ('System Node Temperature', 'Mixed Air Node 1'): "MA Temp.",
    ('Indoor Air Temperature Setpoint', '*'): "Indoor Temp. Setpoint",
    # Comfort
    ('Building Mean PPD', '*'): "PPD",
    ('Zone Thermal Comfort Fanger Model PPD', 'SPACE1-1 People 1'): "PPD Zone 1",
    ('Zone Thermal Comfort Fanger Model PPD', 'SPACE2-1 People 1'): "PPD Zone 2",
    ('Zone Thermal Comfort Fanger Model PPD', 'SPACE3-1 People 1'): "PPD Zone 3",
    ('Zone Thermal Comfort Fanger Model PPD', 'SPACE4-1 People 1'): "PPD Zone 4",
    ('Zone Thermal Comfort Fanger Model PPD', 'SPACE5-1 People 1'): "PPD Zone 5",
    ('Zone Thermal Comfort Fanger Model PMV', 'SPACE1-1 People 1'): "PMV Zone 1",
    ('Zone Thermal Comfort Fanger Model PMV', 'SPACE2-1 People 1'): "PMV Zone 2",
    ('Zone Thermal Comfort Fanger Model PMV', 'SPACE3-1 People 1'): "PMV Zone 3",
    ('Zone Thermal Comfort Fanger Model PMV', 'SPACE4-1 People 1'): "PMV Zone 4",
    ('Zone Thermal Comfort Fanger Model PMV', 'SPACE5-1 People 1'): "PMV Zone 5",
    # Zone Temps
    ('Building Mean Temperature', '*'): "Indoor Temp.",
    ('Zone Air Temperature', 'SPACE1-1'): "Temp Zone 1",
    ('Zone Air Temperature', 'SPACE2-1'): "Temp Zone 2",
    ('Zone Air Temperature', 'SPACE3-1'): "Temp Zone 3",
    ('Zone Air Temperature', 'SPACE4-1'): "Temp Zone 4",
    ('Zone Air Temperature', 'SPACE5-1'): "Temp Zone 5",
    # Lights
    ('Lights Electric Power', 'SPACE1-1 Lights 1'): 'Lights Zone 1',
    ('Lights Electric Power', 'SPACE2-1 Lights 1'): 'Lights Zone 2',
    ('Lights Electric Power', 'SPACE3-1 Lights 1'): 'Lights Zone 3',
    ('Lights Electric Power', 'SPACE4-1 Lights 1'): 'Lights Zone 4',
    ('Lights Electric Power', 'SPACE5-1 Lights 1'): 'Lights Zone 5',
    # Blinds
    ('Surface Shading Device Is On Time Fraction', 'WF-1'): 'Shade On Zone 1',
    ('Surface Window Blind Slat Angle', 'WF-1'): 'Blind Angle Zone 1',
    ('Surface Window Blind Slat Angle', 'WR-1'): 'Blind Angle Zone 2',
    ('Surface Window Blind Slat Angle', 'WB-1'): 'Blind Angle Zone 3',
    ('Surface Window Blind Slat Angle', 'WL-1'): 'Blind Angle Zone 4',
    # Daylighting
    ('Daylighting Reference Point 1 Illuminance', 'SPACE1-1 DaylightingControls'): 'Illuminance 1',
    ('Daylighting Reference Point 1 Daylight Illuminance Setpoint Exceeded Time',
     'SPACE1-1 DaylightingControls'): 'Illuminance Exceeded 1',
    ('Daylighting Reference Point 1 Glare Index', 'SPACE1-1 DaylightingControls'): 'Glare 1',
    ('Daylighting Reference Point 1 Glare Index Setpoint Exceeded Time',
     'SPACE1-1 DaylightingControls'): 'Glare Exceeded 1',

    ('Daylighting Reference Point 2 Illuminance', 'SPACE1-1 DaylightingControls'): 'Illuminance 2',
    ('Daylighting Reference Point 2 Daylight Illuminance Setpoint Exceeded Time',
     'SPACE1-1 DaylightingControls'): 'Illuminance Exceeded 2',
    ('Daylighting Reference Point 2 Glare Index', 'SPACE1-1 DaylightingControls'): 'Glare 2',
    ('Daylighting Reference Point 2 Glare Index Setpoint Exceeded Time',
     'SPACE1-1 DaylightingControls'): 'Glare Exceeded 2',
}

eplus_var_types = {
    'Site Outdoor Air Drybulb Temperature': "Environment",
    'Site Diffuse Solar Radiation Rate per Area': "Environment",
    'Site Direct Solar Radiation Rate per Area': "Environment",
    'Building Mean Temperature': "EMS",
    'Facility Total HVAC Electric Demand Power': 'Whole Building',
    'Building Mean PPD': "EMS",
    'Indoor Air Temperature Setpoint': "EMS",
    'Occupancy Flag': "EMS",
}

all_agent_params = {
    'SAC': {
        "num_actions": 1,
        "policy_type": "Gaussian",
        "gamma": 0.99,
        "tau": 0.005,
        "lr": 0.0003,
        "batch_size": 10,
        "hidden_size": 2,
        "updates_per_step": 1,
        "target_update_interval": 1,
        "replay_size": 10000000,
        "cuda": False,
        "step": 300 * 3,
    },
    'SACwBlinds': {
        "num_actions": 2,
        "policy_type": "Gaussian",
        "gamma": 0.99,
        "tau": 0.005,
        "lr": 0.0003,
        "batch_size": 256,
        "hidden_size": 256,
        "updates_per_step": 1,
        "target_update_interval": 1,
        "replay_size": 10000000,
        "cuda": False,
        "step": 300 * 3,
    },
    'DQN': {
        "lr": 0.001,
        "mini_batch": 48,
        "mem_size": 10000000,
        "gamma": 0.95,
        "batch_size": 256,
        "eps_min": 0.1,
        "replace": 48 * 5,
        "eps_dec": 0.001,
        'num_actions': 10,
        "step": 300 * 3,
    },
    'DQNwBlinds': {
        "lr": 0.001,
        "mini_batch": 48,
        "mem_size": 10000000,
        "gamma": 0.95,
        "batch_size": 256,
        "hidden_size": 256,
        "eps_min": 0.1,
        "replace": 48 * 5,
        "eps_dec": 0.001,
        'num_stpt_actions': 10,
        'num_blind_actions': 10,
        "step": 300 * 3,
    },
    'DDQN': {
        "lr": 0.001,
        "mini_batch": 48,
        "mem_size": 10000000,
        "gamma": 0.95,
        "batch_size": 256,
        "eps_min": 0.1,
        "replace": 48 * 5,
        "eps_dec": 0.001,
        'num_actions': 10,
        "step": 300 * 3,
    },
    'DDQNwBlinds': {
        "lr": 0.001,
        "mini_batch": 48,
        "mem_size": 10000000,
        "gamma": 0.95,
        "batch_size": 256,
        "hidden_size": 256,
        "eps_min": 0.1,
        "replace": 48 * 5,
        "eps_dec": 0.001,
        'num_stpt_actions': 10,
        'num_blind_actions': 10,
        "step": 300 * 3,
    },
    'DDQN_paper': {
        "lr": 0.0001,
        "mini_batch": 64,
        "mem_size": int(1e6),
        "gamma": 0.99,
        "batch_size": 256,
        "eps_min": 0.1,
        "replace": 1000,
        "eps_dec": 0.001,
        'num_actions': 10,
        "step": 300 * 3,
    },
    'DuelingDQN': {
        "lr": 0.0001,
        "mem_size": int(1e6),
        "gamma": 0.99,
        "batch_size": 64,
        "eps_min": 0.1,
        "replace": 1000,
        "eps_dec": 0.001,
        'num_actions': 66,
    },
    'DuelingDQNwBlinds': {
        "lr": 0.0001,
        "mem_size": int(1e6),
        "gamma": 0.99,
        "batch_size": 64,
        "eps_min": 0.1,
        "replace": 1000,
        "eps_dec": 0.001,
        'num_stpt_actions': 66,
        "num_blind_actions": 33
    },
    'PPO': {
        "gamma": 0.98,
        "update_episode": 1,
        "lr": 5e-4,
        "clip_param": 0.1,
        "step": 300,
        "num_actions": 30,
        "discrete": True
    },
    'PPOwBlinds': {
        "gamma": 0.98,
        "update_episode": 1,
        "lr": 5e-4,
        "clip_param": 0.1,
        "step": 300,
        "num_actions": 30,
        "num_blind_actions": 10,
        "discrete": True
    }
}

agent_map = {
    'SAC': SACAgent,
    'SACwBlinds': SACAgent,
    'DQN': DQNAgent,
    'DQNwBlinds': DQNAgentBlinds,
    'DDQN': DDQNAgent,
    'DDQN_paper': DDQNAgent,
    # 'DDQNwBlinds': DDQNAgentBlinds,
    'DuelingDQNwBlinds': BranchingDuelingDQNAgent,
    'PPO': PPOAgent,
    'PPOwBlinds': PPOAgentBlinds
}

reward_map = {
    'Action': ViolationPActionReward,
    'Coil': ViolationPCoilReward,
    'PPD': PPDPActionReward,
    'OCTO': OctoReward
}

stpt_action = {
    "priority": 0,
    "component_type": "Schedule:Constant",
    "control_type": "Schedule Value",
    "actuator_key": "SAT_SP"
}

blind_action = {
    "priority": 0,
    "component_type": "Schedule:Constant",
    "control_type": "Schedule Value",
    "actuator_key": "WF-1_shading_schedule"
}

therm_action_1 = {
    "priority": 0,
    "component_type": "Zone Temperature Control",
    "control_type": "Heating Setpoint",
    "actuator_key": "SPACE1-1"
}

dqn_network_map = {
    'no_relu': DeepQ,
    'leaky': DeepQLeaky,
    'sequential': DeepQSequential,
}

dqn_blinds_network_map = {
    'no_relu': DeepQ_Blinds,
    'leaky': DeepQLeaky_Blinds,
    'sequential': DeepQSequential_Blinds,
}

sac_network_map = {
    'no_relu': DoubleQ,
    'leaky': DoubleQLeaky,
    'sequential': DoubleQSequential,
}

ppo_network_map = {
    'no_relu': [ActorNetwork, CriticNetwork, "no_relu"],
    'leaky': [ActorNetwork, CriticNetwork, "leaky"],
    'sequential': [ActorNetwork, CriticNetwork, "sequential"]
}

branching_dueling_dqn_network_map = {
    'octo': BranchingDuelingDeepQOcto
}
