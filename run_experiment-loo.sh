#!/bin/bash


# MuJoCo cases
envs[0]="Hopper-v2"
#envs[1]="Walker2d-v2"
#envs[2]="AntTruncatedObs-v2"
#envs[3]="HumanoidTruncatedObs-v2"

# DMC cases
#envs[4]="fish-swim"
#envs[5]="hopper-hop"
#envs[6]="quadruped-run"
#envs[7]="cheetah-run"
#envs[8]="humanoid-run"
#envs[9]="humanoid-stand"
#envs[10]="finger-turn_hard"
#envs[11]="hopper-stand"

# training
echo start training
for env in ${envs[@]}
do

# with Layer norm for 5 seeds (0 -- 4)
nohup python loo-main-TH.py -info LOO -env $env -seed 0 -gpu_id 0 -layer_norm 1 -layer_norm_policy 1 >& /dev/null &
nohup python loo-main-TH.py -info LOO -env $env -seed 1 -gpu_id 1 -layer_norm 1 -layer_norm_policy 1 >& /dev/null &
nohup python loo-main-TH.py -info LOO -env $env -seed 2 -gpu_id 2 -layer_norm 1 -layer_norm_policy 1 >& /dev/null &
nohup python loo-main-TH.py -info LOO -env $env -seed 3 -gpu_id 3 -layer_norm 1 -layer_norm_policy 1 >& /dev/null &

wait
done

