#!/bin/bash


#  Hopper case (1e5)
envs[0]="Hopper-v2"
envs[1]="Walker2d-v2"
envs[2]="AntTruncatedObs-v2"
envs[3]="HumanoidTruncatedObs-v2"


# training (for SAC and REDQ)
echo start training
for env in ${envs[@]}
do


# with Layer norm for 5 seeds (0 -- 4)
nohup python main-TH.py -info reset+ToD -env $env -seed 0 -gpu_id 1 -layer_norm 1 -layer_norm_policy 1 -reset_interval 100000 >& /dev/null &
nohup python main-TH.py -info reset+ToD -env $env -seed 1 -gpu_id 2 -layer_norm 1 -layer_norm_policy 1 -reset_interval 100000 >& /dev/null &
nohup python main-TH.py -info reset+ToD -env $env -seed 2 -gpu_id 3 -layer_norm 1 -layer_norm_policy 1 -reset_interval 100000 >& /dev/null &
nohup python main-TH.py -info reset+ToD -env $env -seed 3 -gpu_id 4 -layer_norm 1 -layer_norm_policy 1 -reset_interval 100000 >& /dev/null &

wait
done





# training (for SAC and REDQ)
echo start training
for env in ${envs[@]}
do

# with Layer norm for 5 seeds (0 -- 4)
nohup python main-TH.py -info DroQ+ToD -env $env -seed 0 -gpu_id 1 -layer_norm 1 -layer_norm_policy 1 -target_drop_rate 0.01 >& /dev/null &
nohup python main-TH.py -info DroQ+ToD -env $env -seed 1 -gpu_id 2 -layer_norm 1 -layer_norm_policy 1 -target_drop_rate 0.01 >& /dev/null &
nohup python main-TH.py -info DroQ+ToD -env $env -seed 2 -gpu_id 3 -layer_norm 1 -layer_norm_policy 1 -target_drop_rate 0.01 >& /dev/null &
nohup python main-TH.py -info DroQ+ToD -env $env -seed 3 -gpu_id 4 -layer_norm 1 -layer_norm_policy 1 -target_drop_rate 0.01 >& /dev/null &


wait
done




