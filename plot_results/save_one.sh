#!/bin/bash

# WARN: the order of the loaded modules matters
#module load gentoo/2020 StdEnv/2020
#module load energyplus/9.3.0

module load python/3.6
source ~/hvac_env/bin/activate
#module load python/3.6
# module load openmpi/4.0.3
# module load gentoo/2020 StdEnv/2020

# export OMP_NUM_THREADS=1

# load the virtual environment
# source ~/hvac_env/bin/activate

echo "prog started at: `date`"

srun python save.py --rp1 $power_mult --rp2 $therm_mult --rp3 $vis_mult --a $agent --b $blinds --d $dlights --s $season
