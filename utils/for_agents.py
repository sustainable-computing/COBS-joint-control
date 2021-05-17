import math

import numpy as np
import torch


def augment_ma(observation, action):
    """
    # The RL agent controls the difference between Supply Air Temp.
    and Mixed Air Temp., i.e. the amount of heating from the heating coil.
    But, the E+ expects Supply Air Temp. Setpoint... augment action to account for this
    """
    (last_state, obs_dict, cur_time) = observation
    # if action < 0:
    #     action = torch.zeros_like(action)

    SAT_stpt = obs_dict["MA Temp."] + action  # max(0, action)

    # If the room gets too warm during occupied period, uses outdoor air for free cooling.
    # if (obs_dict["Indoor Temp."] > obs_dict["Indoor Temp. Setpoint"]) & (obs_dict["Occupancy Flag"] == 1):
    #     SAT_stpt = obs_dict["Outdoor Temp."]

    return action, np.array([SAT_stpt])


# def augment_ma_deadband(observation, observation_, action, last_action):
#     """
#     # The RL agent controls the difference between Supply Air Temp.
#     and Mixed Air Temp., i.e. the amount of heating from the heating coil.
#     But, the E+ expects Supply Air Temp. Setpoint... augment action to account for this
#
#     Set a deadband so the agent doesn't occilate back and forth between hot and cold
#     """
#     (last_state, obs_dict, cur_time) = observation
#     (last_state_, obs_dict_, cur_time_) = observation_
#     # if action < 0:
#     #     action = torch.zeros_like(action)
#
#     if (obs_dict_ == None) or (last_action == None):
#         SAT_stpt = obs_dict["MA Temp."] + action
#         return action, np.array([SAT_stpt])
#
#         # Based on EPlus DeadBand Docs
#     # https://bigladdersoftware.com/epx/docs/9-3/input-output-reference/group-zone-controls-thermostats.html#zonecontrolthermostat
#     MAT = obs_dict_['Indoor Temp.']
#     stpt_heat = obs_dict['Heating Setpoint']
#     stpt_cool = obs_dict['Cooling Setpoint']
#
#     SAT_stpt = obs_dict["MA Temp."] + action
#     if (MAT < stpt_heat):
#
#     elif (MAT > stpt_cool):
#         SAT_stpt = last_action
#     else:
#         SAT_stpt = SAT_stpt  # max(0, action)
#
#     # If the room gets too warm during occupied period, uses outdoor air for free cooling.
#     # if (obs_dict["Indoor Temp."] > obs_dict["Indoor Temp. Setpoint"]) & (obs_dict["Occupancy Flag"] == 1):
#     #     SAT_stpt = obs_dict["Outdoor Temp."]
#
#     return action, np.array([SAT_stpt])


def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p


def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
