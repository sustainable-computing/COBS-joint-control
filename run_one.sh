#!/usr/bin/env bash

AGENT_ARGS='--agent_type SAC --network leaky --automatic_entropy_tuning False'
CASE_ARGS='--daylighting False --season heating --blinds False'
REWARD_ARGS='--power_mult 0.1 --therm_mult 1 --vis_mult 0.5 --reward_type OCTO'

python main.py $AGENT_ARGS $CASE_ARGS $REWARD_ARGS --end_run 20 --control_type SAT_SP --control_blinds_multi True