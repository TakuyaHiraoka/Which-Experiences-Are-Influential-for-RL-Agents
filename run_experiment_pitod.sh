#!/bin/bash

# Methods
methods[0]="sac"

# training (for SAC version of PIToD)
echo start training
for method in ${methods[@]}
do

# with Layer norm for 5 seeds (0 -- 4)
nohup python main-TH.py -info $method+ToD${tr}+LN -env Hopper-v2 -seed 0 -gpu_id 0 -method $method -layer_norm 1 -layer_norm_policy 1 >& /dev/null &
nohup python main-TH.py -info $method+ToD${tr}+LN -env Hopper-v2 -seed 1 -gpu_id 1 -method $method -layer_norm 1 -layer_norm_policy 1 >& /dev/null &

nohup python main-TH.py -info $method+ToD${tr}+LN -env Walker2d-v2 -seed 0 -gpu_id 2 -method $method -layer_norm 1 -layer_norm_policy 1 >& /dev/null &
nohup python main-TH.py -info $method+ToD${tr}+LN -env Walker2d-v2 -seed 1 -gpu_id 3 -method $method -layer_norm 1 -layer_norm_policy 1 >& /dev/null &

nohup python main-TH.py -info $method+ToD${tr}+LN -env AntTruncatedObs-v2 -seed 0 -gpu_id 4 -method $method -layer_norm 1 -layer_norm_policy 1 >& /dev/null &
nohup python main-TH.py -info $method+ToD${tr}+LN -env AntTruncatedObs-v2 -seed 1 -gpu_id 5 -method $method -layer_norm 1 -layer_norm_policy 1 >& /dev/null &

nohup python main-TH.py -info $method+ToD${tr}+LN -env HumanoidTruncatedObs-v2 -seed 0 -gpu_id 6 -method $method -layer_norm 1 -layer_norm_policy 1 >& /dev/null &
nohup python main-TH.py -info $method+ToD${tr}+LN -env HumanoidTruncatedObs-v2 -seed 1 -gpu_id 7 -method $method -layer_norm 1 -layer_norm_policy 1 >& /dev/null &

wait
done


