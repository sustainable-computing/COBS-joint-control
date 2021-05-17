#!/usr/bin/bash

AGENTS=('DuelingDQN')
BLINDS=('False')
DLIGHT=('True' 'False')
SEASON=('heating' 'cooling')

POWER_MULT=(0.1 0.4 0.7 1)
THERM_MULT=(0.1 0.4 0.7 1)
VIS_MULT=(0.1 0.4 0.7 1)

for agent in ${AGENTS[@]} ; do
    for blinds in ${BLINDS[@]} ; do
	for dlights in ${DLIGHT[@]} ; do
	    for season in ${SEASON[@]} ; do
	        for power_mult in ${POWER_MULT[@]} ; do
	            for therm_mult in ${THERM_MULT[@]} ; do
	                for vis_mult in ${VIS_MULT[@]} ; do
		        export power_mult therm_mult vis_mult agent blinds dlights season
 	   	        sbatch save_one.sh
  	  	        done
		    done
	        done
	    done
        done
    done
done
