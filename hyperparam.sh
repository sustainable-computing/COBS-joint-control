#!/usr/bin/bash

POWER_MULT=(0.1 0.4 0.7 1)
THERM_MULT=(0.1 0.4 0.7 1)
VIS_MULT=(0.1 0.4 0.7 1)

occupancy=True
agent=PPO
network=leaky
#POWER_MULT=(0.1)
#THERM_MULT=(0.1)
#VIS_MULT=(0.1)

for power_mult in ${POWER_MULT[@]} ; do
  for therm_mult in ${THERM_MULT[@]} ; do
    for vis_mult in ${VIS_MULT[@]} ; do
      export power_mult therm_mult vis_mult occupancy agent network
      sbatch hyperparam_ppo.sh
      #      sbatch hyperparam_dueling.sh
      #       sbatch hyperparam_dqn_one_run.sh
    done
  done
done
