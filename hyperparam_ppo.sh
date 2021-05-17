#!/bin/bash

# WARN: the order of the loaded modules matters
module load StdEnv/2020 energyplus/9.3.0
module load cuda cudnn
# load the virtual environment
source ~/env_py3.7/bin/activate

export OMP_NUM_THREADS=1

echo "prog started at: `date`"

case $SLURM_ARRAY_TASK_ID in
    0) ARGS="--daylighting True --season heating --blinds False --control_blinds_multi False";;
    1) ARGS="--daylighting False --season heating --blinds False --control_blinds_multi False";;
    2) ARGS="--daylighting True --season heating --blinds True --control_blinds_multi False";;
    3) ARGS="--daylighting False --season heating --blinds True --control_blinds_multi False";;
    4) ARGS="--daylighting True --season heating --blinds True --control_blinds_multi True";;
    5) ARGS="--daylighting False --season heating --blinds True --control_blinds_multi True";;
    6) ARGS="--daylighting True --season cooling --blinds False --control_blinds_multi False";;
    7) ARGS="--daylighting False --season cooling --blinds False --control_blinds_multi False";;
    8) ARGS="--daylighting True --season cooling --blinds True --control_blinds_multi False";;
    9) ARGS="--daylighting False --season cooling --blinds True --control_blinds_multi False";;
    10) ARGS="--daylighting True --season cooling --blinds True --control_blinds_multi True";;
    11) ARGS="--daylighting False --season cooling --blinds True --control_blinds_multi True";;
esac

python3 main.py $ARGS --random_occupancy $occupancy --vav False --multi_agent False --network $network --control_sat True --load_sat False --load_sat_path '<REMOVED>' --control_therm False --power_mult $power_mult --therm_mult $therm_mult --vis_mult $vis_mult --save_root '<REMOVED>' --end_run 400 --agent_type $agent --reward_type OCTO --eplus_path '<REMOVED>'