

# What is this?

This is the source code for replicating the experiments reported in ["Which Experiences Are Influential for RL Agents? Efficiently Estimating The Influence of Experiences."](https://openreview.net/forum?id=pUvF97zAu9) [(poster)](https://drive.google.com/file/d/1fqd5UPUNOQniG-CshmdFFPxEG9m7W4hS/view?usp=sharing) [(slides)](https://drive.google.com/file/d/1JjOMvA-oF7bas2OJmO_en6mJAtNGoLjs/view?usp=sharing)

This codebase provides a method to estimate and disable the influence of experiences on the performance of reinforcement learning (RL) agents **WITHOUT** retraining them. 
![Outline](figure_readme/pitod_outline.jpg)


What is this functionality used for? This functionality can be used to debug RL agents. 
When an RL agent fails to learn properly, identifying and disabling the experiences that negatively affect the agent can improve its performance. 
The following videos demonstrate examples of this debugging process: 

https://github.com/user-attachments/assets/07d14535-bf16-4069-893a-f08f9ee9c7c7

https://github.com/user-attachments/assets/a47d8a54-a794-4e04-a48d-05e03ad31e9e


# Requirements
You can install the required libraries using `pip install -r requirements.txt`, except for `mujoco_py`. 

Note that you need a licence to install `mujoco_py`. For installation, please follow instructions [here](https://github.com/openai/mujoco-py).


# Usage
Currently, this codebase supports [SAC](https://proceedings.mlr.press/v80/haarnoja18b.html), [DroQ](https://openreview.net/forum?id=xCVJMsPv3RT), and [reset](https://proceedings.mlr.press/v162/nikishin22a) agents.

To estimate the influence of experiences on a SAC agent, run `main-TH.py`, e.g., 
```
python main-TH.py -info SAC+ToD -env Hopper-v2 -seed 0 -gpu_id 0 -layer_norm 1 -layer_norm_policy 1
```

For DroQ and reset agents, you can specify the arguments `-target_drop_rate` and `-reset_interval` respectively and then run `main-TH.py`, e.g., 
```
python main-TH.py -info DroQ+ToD -env Hopper-v2 -seed 0 -gpu_id 0 -layer_norm 1 -layer_norm_policy 1 -target_drop_rate 0.01
```
```
python main-TH.py -info reset+ToD -env Hopper-v2 -seed 0 -gpu_id 0 -layer_norm 1 -layer_norm_policy 1 -reset_interval 100000
```
Additional examples of execution commands can be found in the scripts `run_experiment-pitod.sh` and `run_experiment-pitod-reset-droq.sh`.

The experimental results will be recorded in the `runs` directory. This folder will contain logs of the agent's learning, evaluation, and the results of experience influence estimation on returns and Q-estimation bias. 

To visualize the results, run the following command:
```
python plot_main_results_pitod.py
```


## Note: PIToD runtime and memory

**Runtime**
In *return* influence estimation, **for each mask** the agent is evaluated **n_eval * 2** times (for flip and non-flip cases) by interacting with an evaluation environment (default: `n_eval=10`). If the environment is heavy, this can dominate wall-clock time. For speedup, reduce `n_eval` (e.g., `10 -> 1`). 
Also, increasing `experience_group_size` reduces the number of experience-group targets to evaluate (e.g., `5000 -> 10000`), often providing a substantial speedup. 

**Memory footprint**
- PIToD uses macro dropout (see the appendix of my paper for details) and **larger models than (vanilla) 256 x 256 MLP**, so VRAM usage is higher. On most modern GPUs this should be fine, but if your memory resource is tight, reduce `hidden_sizes` (e.g., `256 -> 80` per layer). 

**Example**
For example, the above can be specified with the following command: 
```
python main-TH.py  -info SAC+ToD -env Hopper-v2 -seed 0 -gpu_id 0 -layer_norm 1 -layer_norm_policy 1 -n_eval 1 -experience_group_size 10000 -hidden_sizes 80 80
```


# Leave-One-Out (LOO) method

We also provide a simple implementation of the Leave-One-Out (LOO) method to estimate the influence of experiences.
To use LOO, run loo-main-TH.py, e.g.,  
```
python loo-main-TH.py -info LOO -env Hopper-v2 -seed 0 -gpu_id 0 -layer_norm 1 -layer_norm_policy 1
```

Additional examples of execution commands can be found in the script `run_experiment-loo.sh`. 
The experimental results will be recorded in `runs/..../baseline`. 



# Citation
If you use this repo or find it useful, please consider citing:
```
@inproceedings{hiraoka2025which,
title={Which Experiences Are Influential for {RL} Agents? Efficiently Estimating The Influence of Experiences},
author={Takuya Hiraoka and Takashi Onishi and Guanquan Wang and Yoshimasa Tsuruoka},
booktitle={Reinforcement Learning Conference},
year={2025},
url={https://openreview.net/forum?id=pUvF97zAu9}
}
```
