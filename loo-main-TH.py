from typing import Dict, Tuple, Union

import gym
import numpy as np
import torch
import time
import sys

from redq.algos.redq_sac import REDQSACAgent
from redq.algos.core import mbpo_epoches, test_agent
from redq.utils.run_utils import setup_logger_kwargs
from redq.utils.bias_utils import log_evaluation
from redq.utils.logx import EpochLogger

from pathlib import Path
from redq.algos.core import ReplayBuffer
from tqdm import tqdm

import customenvs
import dmc2gym
# register environments with truncated observations
customenvs.register_mbpo_environments()
# DM control suite
dm_control_env = ["fish-swim", "hopper-hop", "quadruped-run",
                  "cheetah-run", "humanoid-run", "humanoid-stand",
                  "finger-turn_hard", "hopper-stand"]


def loo(env_name: str,
          seed: int = 0,
          epochs: int = -1,
          steps_per_epoch: int = 5000,
          max_ep_len: int = 1000,
          n_evals_per_epoch: int = 10,
          adversarial_reward_epoch: int = -999,
          logger_kwargs: Dict = dict(),
          gpu_id: int = 0,
          # The following are base agent related hyperparameters
          hidden_sizes: Tuple[int, ...] = (int(256 / 2), int(256 / 2)),
          replay_size: int = int(2e6),
          batch_size: int = 256,
          lr: float = 3e-4,
          gamma: float = 0.99,
          polyak: float = 0.995,
          alpha: float = 0.2,
          auto_alpha: bool = True,
          target_entropy: Union[str, float] = 'mbpo',
          start_steps: int = 5000,
          delay_update_steps: Union[str, int] = 'auto',
          utd_ratio: int = 4,
          num_Q: int = 2,
          num_min: int = 2,
          policy_update_delay: int = 20,
          # The following are bias evaluation related
          evaluate_bias: bool = True,
          n_mc_eval: int = 1000,
          n_mc_cutoff: int = 350,
          reseed_each_epoch: bool = True,
          # The following are PIToD network structure related
          layer_norm: bool = False,
          layer_norm_policy: bool = False,
          experience_group_size: int = 5000,
          mask_dim: int = 20,
          target_drop_rate: float = 0.0,
          reset_interval: int = -1,
          # The following are PIToD evaluation related
          experience_cleansing: bool = True,
          dump_trajectory_for_demo: bool = False,  # True,
          record_training_self_training_losses: bool = True,
          influence_estimation_interval: int = -999999, # don't estimate influence in LOO baseline
          ):
    """
    Run LOO algorithm.
    
    :param env_name: Name of the gym environment.
    :param seed: Random seed.
    :param epochs: Total number of epochs.
    :param steps_per_epoch: Number of timesteps (i.e., environment interactions) for each epoch.
    :param max_ep_len: Maximum number of timesteps until an episode terminates.
    :param n_evals_per_epoch: Number of evaluations for each epoch.
    :param adversarial_reward_epoch: The epoch during which the agent encounters adversarial rewards.
    :param logger_kwargs: Arguments for the logger.
    :param gpu_id: GPU ID to be used for computation.
    :param hidden_sizes: Sizes of the hidden layers of Q and policy networks.
    :param replay_size: Size of the replay buffer.
    :param batch_size: Mini-batch size.
    :param lr: Learning rate for Q and policy networks.
    :param gamma: Discount factor.
    :param polyak: Hyperparameter for Polyak-averaged target networks.
    :param alpha: SAC entropy hyperparameter.
    :param auto_alpha: Whether to use adaptive entropy coefficient.
    :param target_entropy: Target entropy used for adaptive entropy coefficient.
    :param start_steps: Number of experiences collected at the beginning of training.
    :param delay_update_steps: Number of experiences collected before starting updates.
    :param utd_ratio: Update-to-data (Q and policy network update) ratio.
    :param num_Q: Number of Q networks in the Q ensemble.
    :param num_min: Number of sampled Q values to take the minimum from.
    :param policy_update_delay: Number of updates before updating the policy network.
    :param evaluate_bias: Whether to evaluate Q-estimation bias.
    :param n_mc_eval: The total number of true Q-values used for bias evaluation.
    :param n_mc_cutoff: The cutoff episode length for Q-bias evaluation.
    :param reseed_each_epoch: Whether to reseed the random number generator at the start of each epoch.
    :param layer_norm: Whether to use layer normalization for Q-networks.
    :param layer_norm_policy: Whether to use layer normalization in the policy network.
    :param experience_group_size: size of experience group
    :param mask_dim: size of mask for turn-over dropout.
    :param target_drop_rate: The rate at which each weight of Q-networks is dropped.
    :param reset_interval: Interval (number of environment interactions) for periodical parameter reset.
    :param experience_cleansing: Periodical evaluation with deletion of negatively influential experiences.
    :param dump_trajectory_for_demo: Whether to dump the trajectory for visualization purposes.
    :param record_training_self_training_losses: Whether to record training and self-training losses.
    :param influence_estimation_interval:  interval for influence estimation and policy/Q-function amendment.
    """

    # set device to gpu if available
    device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")

    # set number of epoch
    if epochs < 0:
        mbpo_epoches['AntTruncatedObs-v2'] = 300
        mbpo_epoches['HumanoidTruncatedObs-v2'] = 300
        mbpo_epoches.update(dict(zip(dm_control_env, [300 for _ in dm_control_env])))
        epochs = mbpo_epoches[env_name]
    total_steps = steps_per_epoch * epochs + 1

    # set up logger
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # set up environment and seeding
    if args.env in dm_control_env:
        domain_name, task_name = args.env.split("-")[0], args.env.split("-")[1]
        env = dmc2gym.make(domain_name, task_name)
        test_env = dmc2gym.make(domain_name, task_name)
        bias_eval_env = dmc2gym.make(domain_name, task_name)
        if target_entropy == "mbpo":  # change target entropy mode to auto as mbpo is not supported for DMC cases
            target_entropy = 'auto'
    else:
        env, test_env, bias_eval_env = gym.make(args.env), gym.make(args.env), gym.make(args.env)

    # seed torch and numpy
    torch.manual_seed(seed)
    np.random.seed(seed)

    # seed environment along with env action space so that everything is properly seeded for reproducibility
    def seed_all(epoch):
        seed_shift = epoch * 9999
        mod_value = 999999
        env_seed = (seed + seed_shift) % mod_value
        test_env_seed = (seed + 10000 + seed_shift) % mod_value
        bias_eval_env_seed = (seed + 20000 + seed_shift) % mod_value
        torch.manual_seed(env_seed)
        np.random.seed(env_seed)
        env.seed(env_seed)
        env.action_space.np_random.seed(env_seed)
        test_env.seed(test_env_seed)
        test_env.action_space.np_random.seed(test_env_seed)
        bias_eval_env.seed(bias_eval_env_seed)
        bias_eval_env.action_space.np_random.seed(bias_eval_env_seed)

    seed_all(epoch=0)

    # prepare to init agent
    # get obs and action dimensions
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    # if environment has a smaller max episode length, then use the environment's max episode length
    max_ep_len = env._max_episode_steps if max_ep_len > env._max_episode_steps else max_ep_len
    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    # we need .item() to convert it from numpy float to python float
    act_limit = env.action_space.high[0].item()
    # keep track of run time
    start_time = time.time()
    # flush logger (optional)
    sys.stdout.flush()

    # init agent and start training
    agent = REDQSACAgent(env_name=env_name, obs_dim=obs_dim, act_dim=act_dim, act_limit=act_limit, device=device,
                         hidden_sizes=hidden_sizes, replay_size=replay_size, batch_size=batch_size,
                         lr=lr, gamma=gamma, polyak=polyak, alpha=alpha, auto_alpha=auto_alpha,
                         target_entropy=target_entropy, start_steps=start_steps, delay_update_steps=delay_update_steps,
                         utd_ratio=utd_ratio, num_Q=num_Q, num_min=num_min, policy_update_delay=policy_update_delay,
                         target_drop_rate=target_drop_rate, layer_norm=layer_norm, layer_norm_policy=layer_norm_policy,
                         experience_group_size=experience_group_size, mask_dim=mask_dim,
                         )

    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

    for t in range(total_steps):
        # get action from agent
        a = agent.get_exploration_action(o, env)
        # step the env, get next observation, reward and done signal
        o2, r, d, _ = env.step(a)

        if ((t >= (adversarial_reward_epoch * steps_per_epoch))
                and (t <= (adversarial_reward_epoch * steps_per_epoch + steps_per_epoch))):
            r = r * (-100.0)

        # Very important: before we let agent store this transition,
        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        ep_len += 1
        d = False if ep_len == max_ep_len else d

        # give new data to agent
        agent.store_data(o, a, r, o2, d)
        # let agent update
        agent.train(logger)
        # set obs to next obs
        o = o2
        ep_ret += r

        if d or (ep_len == max_ep_len):
            # store episode return and length to logger
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            # reset environment
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        # End of epoch wrap-up
        if (t + 1) % steps_per_epoch == 0:
            epoch = t // steps_per_epoch

            # LOO influence estimation
            # Create baseline logger (only once)
            if epoch == 50 :
                print("(approximated) LOO evaluation")
                if 'baseline_logger' not in locals():
                    bl_kwargs = logger_kwargs.copy()
                    out_dir = Path(bl_kwargs.get('output_dir', './')) / 'baseline'
                    bl_kwargs['output_dir'] = str(out_dir)
                    baseline_logger = EpochLogger(**bl_kwargs)
                    baseline_logger.save_config({'note': 'baseline evaluations'})

                # Compute number of experience groups
                num_groups = agent.replay_buffer.size // experience_group_size
                # Determine how many sliding windows to run
                num_windows = max(1, num_groups // 1)

                for i in tqdm(range(num_windows)):
                    # 1) Determine exclusion range
                    excl_start = experience_group_size * 1 * i
                    excl_end = min(excl_start + experience_group_size * 1, agent.replay_buffer.size)

                    # Generate mask for indices to keep
                    idx_all = np.arange(agent.replay_buffer.size)
                    keep_mask = (idx_all < excl_start) | (idx_all >= excl_end)
                    keep_idxs = idx_all[keep_mask]

                    # 2) Create a new replay buffer and copy the kept experiences
                    rb_copy = ReplayBuffer(obs_dim, act_dim, replay_size,
                                           experience_group_size=experience_group_size,
                                           mask_dim=mask_dim)

                    for idx in keep_idxs:
                        rb_copy.store(agent.replay_buffer.obs1_buf[idx],
                                      agent.replay_buffer.acts_buf[idx],
                                      agent.replay_buffer.rews_buf[idx],
                                      agent.replay_buffer.obs2_buf[idx],
                                      agent.replay_buffer.done_buf[idx])

                    # 3) Create a deepcopy of the agent and assign the copied buffer
                    agent_copy = REDQSACAgent(env_name=env_name, obs_dim=obs_dim, act_dim=act_dim, act_limit=act_limit, device=device,
                             hidden_sizes=hidden_sizes, replay_size=replay_size, batch_size=batch_size,
                             lr=lr, gamma=gamma, polyak=polyak, alpha=alpha, auto_alpha=auto_alpha,
                             target_entropy=target_entropy, start_steps=start_steps, delay_update_steps=delay_update_steps,
                             utd_ratio=utd_ratio, num_Q=num_Q, num_min=num_min, policy_update_delay=policy_update_delay,
                             target_drop_rate=target_drop_rate, layer_norm=layer_norm, layer_norm_policy=layer_norm_policy,
                             experience_group_size=experience_group_size, mask_dim=mask_dim,
                             )
                    agent_copy.replay_buffer = rb_copy

                    # do offline training
                    for _ in range(75000):
                        agent_copy.train(baseline_logger)

                    # 4) Evaluate and log performance of the modified agent
                    test_agent(agent_copy, test_env, max_ep_len, baseline_logger,
                               n_eval=n_evals_per_epoch)
                    log_evaluation(bias_eval_env, agent_copy, baseline_logger, max_ep_len, alpha, gamma, n_mc_eval, n_mc_cutoff,
                                        experience_cleansing=experience_cleansing,
                                        dump_trajectory_for_demo=dump_trajectory_for_demo,
                                        record_training_self_training_losses=record_training_self_training_losses,
                                        influence_estimation_interval=influence_estimation_interval,
                                        )
                    baseline_logger.log_tabular('Epoch', epoch)
                    baseline_logger.log_tabular('TotalEnvInteracts', t)
                    baseline_logger.log_tabular('Time', time.time() - start_time)
                    baseline_logger.log_tabular('TestEpRet', with_min_and_max=True)
                    baseline_logger.log_tabular('TestEpLen', average_only=True)
                    baseline_logger.log_tabular('Q1Vals', with_min_and_max=True)
                    baseline_logger.log_tabular('LossQ1')
                    baseline_logger.log_tabular('LogPi', with_min_and_max=True)
                    baseline_logger.log_tabular('LossPi', average_only=True)
                    baseline_logger.log_tabular('Alpha', with_min_and_max=True)
                    baseline_logger.log_tabular('LossAlpha', average_only=True)
                    baseline_logger.log_tabular('PreTanh', with_min_and_max=True)
                    baseline_logger.log_tabular("MCDisRet", with_min_and_max=True)
                    baseline_logger.log_tabular("MCDisRetEnt", with_min_and_max=True)
                    baseline_logger.log_tabular("QPred", with_min_and_max=True)
                    baseline_logger.log_tabular("QBias", with_min_and_max=True)
                    baseline_logger.log_tabular("QBiasAbs", with_min_and_max=True)
                    baseline_logger.log_tabular("NormQBias", with_min_and_max=True)
                    baseline_logger.log_tabular("QBiasSqr", with_min_and_max=True)
                    baseline_logger.log_tabular("NormQBiasSqr", with_min_and_max=True)
                    baseline_logger.dump_tabular()

                return # do only once


            # Test the performance of the deterministic version of the agent.
            test_agent(agent, test_env, max_ep_len, logger, n_eval=n_evals_per_epoch)  # add logging here
            if evaluate_bias:
                log_evaluation(bias_eval_env, agent, logger, max_ep_len, alpha, gamma, n_mc_eval, n_mc_cutoff,
                                    experience_cleansing=experience_cleansing,
                                    dump_trajectory_for_demo=dump_trajectory_for_demo,
                                    record_training_self_training_losses=record_training_self_training_losses,
                                    influence_estimation_interval=influence_estimation_interval,
                                    )

            # reseed should improve reproducibility (should make results the same whether bias evaluation is on or not)
            if reseed_each_epoch:
                seed_all(epoch)

            # logging
            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Time', time.time() - start_time)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('LossQ1')
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('Alpha', with_min_and_max=True)
            logger.log_tabular('LossAlpha', average_only=True)
            logger.log_tabular('PreTanh', with_min_and_max=True)

            if evaluate_bias:
                logger.log_tabular("MCDisRet", with_min_and_max=True)
                logger.log_tabular("MCDisRetEnt", with_min_and_max=True)
                logger.log_tabular("QPred", with_min_and_max=True)
                logger.log_tabular("QBias", with_min_and_max=True)
                logger.log_tabular("QBiasAbs", with_min_and_max=True)
                logger.log_tabular("NormQBias", with_min_and_max=True)
                logger.log_tabular("QBiasSqr", with_min_and_max=True)
                logger.log_tabular("NormQBiasSqr", with_min_and_max=True)

            logger.dump_tabular()

            # flush logged information to disk
            sys.stdout.flush()

        # ResetToD
        if ((t % reset_interval) == 0) and (reset_interval >= 0):
            agent.reset()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-env', type=str, default='Hopper-v2',
                        help="Name of the gym environment. Default is Hopper-v2.")
    parser.add_argument('-seed', type=int, default=0,
                        help="Random seed. Default is 0.")
    parser.add_argument('-epochs', type=int, default=-1,
                        help="Number of epochs. Default is -1, "
                             "which means using the epochs specified in the MBPO paper.")
    parser.add_argument('-exp_name', type=str, default='redq_sac',
                        help="Name of the experiment. Default is redq_sac.")
    parser.add_argument('-data_dir', type=str, default='./data/',
                        help="Directory to save data. Default is ./data/.")
    parser.add_argument("-info", type=str,
                        help="Name of the run. Generally set to the name of the RL method (e.g., SAC+ToD).")
    parser.add_argument("-gpu_id", type=int, default=0,
                        help="GPU device ID to be used in experiment with GPU. Default is 0.")
    parser.add_argument("-target_drop_rate", type=float, default=0.0,
                        help="Dropout rate for the Q-network. Default is 0.")
    parser.add_argument("-layer_norm", type=int, default=0, choices=[0, 1],
                        help="Use layer normalization for the Q-network if set to 1. Default is 0 (False).")
    parser.add_argument("-num_q", type=int, default=2,
                        help="Number of Q networks in the Q ensemble. Default is 2.")
    parser.add_argument("-layer_norm_policy", type=int, default=0, choices=[0, 1],
                        help="Use layer normalization for the policy network if set to 1. Default is 0 (False).")
    parser.add_argument("-reset_interval", type=int, default=-1,
                        help="Reset interval w.r.t the number of environment interactions. "
                             "Default is -1, which means no reset.")
    parser.add_argument("-adversarial_reward_epoch", type=int, default=-999,
                        help="The epoch during which the agent encounters adversarial rewards. Default is -999.")

    args = parser.parse_args()

    # setup experiment log directories
    args.data_dir = './runs/' + str(args.info) + '/'
    exp_name_full = args.exp_name + '_%s' % args.env
    # - specify experiment name, seed and data_dir.
    # - for example, for seed 0, the progress.txt will be saved under runs/data_dir/exp_name/exp_name_s0
    logger_kwargs = setup_logger_kwargs(exp_name_full, args.seed, args.data_dir)

    loo(args.env, seed=args.seed, epochs=args.epochs, logger_kwargs=logger_kwargs, gpu_id=args.gpu_id,
          num_Q=args.num_q, layer_norm=bool(args.layer_norm), layer_norm_policy=bool(args.layer_norm_policy),
          target_drop_rate=args.target_drop_rate, reset_interval=args.reset_interval,
          adversarial_reward_epoch=args.adversarial_reward_epoch)
