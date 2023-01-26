import numpy as np
import torch
from torch import Tensor

import pickle
import bz2

def get_mc_return_with_entropy_on_reset(bias_eval_env, agent, max_ep_len, alpha, gamma, n_mc_eval, n_mc_cutoff):
    # since we want to also compute bias, so we need to
    final_mc_list = np.zeros(0)
    final_mc_entropy_list = np.zeros(0)
    final_obs_list = []
    final_act_list = []
    while final_mc_list.shape[0] < n_mc_eval:
        # we continue if haven't collected enough data
        o = bias_eval_env.reset()
        # temporary lists
        reward_list, log_prob_a_tilda_list, obs_list, act_list = [], [], [], []
        r, d, ep_ret, ep_len = 0, False, 0, 0
        discounted_return = 0
        discounted_return_with_entropy = 0
        for i_step in range(max_ep_len):  # run an episode
            with torch.no_grad():
                a, log_prob_a_tilda = agent.get_action_and_logprob_for_bias_evaluation(o)
            obs_list.append(o)
            act_list.append(a)
            o, r, d, _ = bias_eval_env.step(a)
            ep_ret += r
            ep_len += 1
            reward_list.append(r)
            log_prob_a_tilda_list.append(log_prob_a_tilda.item())
            if d or (ep_len == max_ep_len):
                break
        discounted_return_list = np.zeros(ep_len)
        discounted_return_with_entropy_list = np.zeros(ep_len)
        for i_step in range(ep_len - 1, -1, -1):
            # backwards compute discounted return and with entropy for all s-a visited
            if i_step == ep_len - 1:
                discounted_return_list[i_step] = reward_list[i_step]
                discounted_return_with_entropy_list[i_step] = reward_list[i_step]
            else:
                discounted_return_list[i_step] = reward_list[i_step] + gamma * discounted_return_list[i_step + 1]
                discounted_return_with_entropy_list[i_step] = reward_list[i_step] + \
                                                              gamma * (discounted_return_with_entropy_list[i_step + 1] - alpha * log_prob_a_tilda_list[i_step + 1])
        # now we take the first few of these.
        final_mc_list = np.concatenate((final_mc_list, discounted_return_list[:n_mc_cutoff]))
        final_mc_entropy_list = np.concatenate(
            (final_mc_entropy_list, discounted_return_with_entropy_list[:n_mc_cutoff]))
        final_obs_list += obs_list[:n_mc_cutoff]
        final_act_list += act_list[:n_mc_cutoff]
    return final_mc_list, final_mc_entropy_list, np.array(final_obs_list), np.array(final_act_list)

def log_bias_evaluation(bias_eval_env, agent, logger, max_ep_len, alpha, gamma, n_mc_eval, n_mc_cutoff):
    final_mc_list, final_mc_entropy_list, final_obs_list, final_act_list = get_mc_return_with_entropy_on_reset(bias_eval_env, agent, max_ep_len, alpha, gamma, n_mc_eval, n_mc_cutoff)
    logger.store(MCDisRet=final_mc_list)
    logger.store(MCDisRetEnt=final_mc_entropy_list)
    obs_tensor = Tensor(final_obs_list).to(agent.device)
    acts_tensor = Tensor(final_act_list).to(agent.device)
    with torch.no_grad():
        q_prediction = agent.get_ave_q_prediction_for_bias_evaluation(obs_tensor, acts_tensor).cpu().numpy().reshape(-1)
    bias = q_prediction - final_mc_entropy_list
    bias_abs = np.abs(bias)
    bias_squared = bias ** 2
    logger.store(QPred=q_prediction)
    logger.store(QBias=bias)
    logger.store(QBiasAbs=bias_abs)
    logger.store(QBiasSqr=bias_squared)
    final_mc_entropy_list_normalize_base = final_mc_entropy_list.copy()
    final_mc_entropy_list_normalize_base = np.abs(final_mc_entropy_list_normalize_base)
    final_mc_entropy_list_normalize_base[final_mc_entropy_list_normalize_base < 10] = 10
    normalized_bias_per_state = bias / final_mc_entropy_list_normalize_base
    logger.store(NormQBias=normalized_bias_per_state)
    normalized_bias_sqr_per_state = bias_squared / final_mc_entropy_list_normalize_base
    logger.store(NormQBiasSqr=normalized_bias_sqr_per_state)


    # estimate data influence on overestimation bias. TH 20221026
    sample_mask_size = 256
    eval_data_size = bias.size
    with torch.no_grad():
        # - generate indices by uniform sampling ranging from 0 to agent.replay_buffer.unique_id - 1,
        indices = torch.range(start=0,
                              end=(agent.replay_buffer.unique_id - 1),
                              step=int((agent.replay_buffer.unique_id - 1) / sample_mask_size)).reshape((-1, 1)).to(agent.device)
        indices = torch.unsqueeze(indices.repeat(1, eval_data_size), -1) # eval_data_size x  x 1
        if not hasattr(agent, "is_init_histories"):

            agent.eval_history_flip_biases = [] # TH20221030
            agent.eval_history_non_flip_biases = []  # TH202210331

            agent.is_init_histories = True

        # - estimate value non-flip, flip for each element of index.
        influence_biases = [] 
        flip_biases = []
        non_flip_biases = []
        for i in range(sample_mask_size):
            i_indices = indices[i, :, :]

            i_q_prediction_non_flipped = agent.get_ave_q_prediction_for_bias_evaluation(obs_tensor, acts_tensor, indices=i_indices, flips=False).cpu().numpy().reshape(-1)
            i_q_prediction_flipped = agent.get_ave_q_prediction_for_bias_evaluation(obs_tensor, acts_tensor, indices=i_indices, flips=True).cpu().numpy().reshape(-1)

            # compute bias with flipped mask (i.e., param wo influence of ith data) TH20221030
            flip_bias = np.mean( (i_q_prediction_flipped - final_mc_entropy_list) / final_mc_entropy_list_normalize_base)
            flip_biases.append(flip_bias)

            # compute bias with non-flipped mask TH20221031
            non_flip_bias = np.mean( (i_q_prediction_non_flipped - final_mc_entropy_list) / final_mc_entropy_list_normalize_base)
            non_flip_biases.append(non_flip_bias)

        # flip bias TH20221030
        agent.eval_history_flip_biases.append(flip_biases)
        # - dump result (aa compressed pickle file)
        pkl = pickle.dumps(agent.eval_history_flip_biases)
        fout = bz2.BZ2File(logger.output_dir + "/eval_history_flip_biases.bz2", "wb", compresslevel=9)
        fout.write(pkl)
        fout.close()
        # non-flip bias TH20221031
        agent.eval_history_non_flip_biases.append(non_flip_biases)
        # - dump result (aa compressed pickle file)
        pkl = pickle.dumps(agent.eval_history_non_flip_biases)
        fout = bz2.BZ2File(logger.output_dir + "/eval_history_non_flip_biases.bz2", "wb", compresslevel=9)
        fout.write(pkl)
        fout.close()
