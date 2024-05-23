import numpy as np
import torch
from torch import Tensor

import pickle
import bz2

import tqdm

def get_mc_return_with_entropy_on_reset(bias_eval_env, agent, max_ep_len, alpha, gamma, n_mc_eval, n_mc_cutoff):
    # since we want to also compute bias, so we need to
    final_mc_list = np.zeros(0)
    final_mc_entropy_list = np.zeros(0)
    final_obs_list = []
    final_act_list = []
    final_done_list = []  # add for demo. TH20230217

    while final_mc_list.shape[0] < n_mc_eval:
        # we continue if agent haven't collected enough data
        o = bias_eval_env.reset()
        # temporary lists
        reward_list, log_prob_a_tilda_list, obs_list, act_list = [], [], [], []
        done_list = []  # TH20230217
        r, d, ep_ret, ep_len = 0, False, 0, 0
        discounted_return = 0
        discounted_return_with_entropy = 0
        for i_step in range(max_ep_len):  # run an episode
            with torch.no_grad():
                a, log_prob_a_tilda = agent.get_action_and_logprob_for_bias_evaluation(o)
            obs_list.append(o)
            act_list.append(a)
            o, r, d, _ = bias_eval_env.step(a)
            done_list.append(d)  # TH20230217
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
                                                              gamma * (discounted_return_with_entropy_list[
                                                                           i_step + 1] - alpha * log_prob_a_tilda_list[
                                                                           i_step + 1])
        # now we take the first few of these.
        final_mc_list = np.concatenate((final_mc_list, discounted_return_list[:n_mc_cutoff]))
        final_mc_entropy_list = np.concatenate(
            (final_mc_entropy_list, discounted_return_with_entropy_list[:n_mc_cutoff]))
        final_obs_list += obs_list[:n_mc_cutoff]
        final_act_list += act_list[:n_mc_cutoff]
        final_done_list += done_list[:n_mc_cutoff]
    return final_mc_list, final_mc_entropy_list, np.array(final_obs_list), np.array(final_act_list), np.array(
        final_done_list)


def get_mc_return_with_entropy_and_obs_act(bias_eval_env, agent, max_ep_len, alpha, gamma, n_mc_eval, n_mc_cutoff):
    _, final_mc_entropy_list, final_obs_list, final_act_list, final_done_list = get_mc_return_with_entropy_on_reset(
        bias_eval_env, agent, max_ep_len, alpha, gamma, n_mc_eval, n_mc_cutoff)
    obs_tensor = Tensor(final_obs_list).to(agent.device)
    acts_tensor = Tensor(final_act_list).to(agent.device)
    final_mc_entropy_list_normalize_base = final_mc_entropy_list.copy()
    final_mc_entropy_list_normalize_base = np.abs(final_mc_entropy_list_normalize_base)
    final_mc_entropy_list_normalize_base[final_mc_entropy_list_normalize_base < 10] = 10

    return final_mc_entropy_list, final_mc_entropy_list_normalize_base, obs_tensor, acts_tensor


def _evaluateL(agent, sample_mask_size, eval_data_size, obs_tensor, acts_tensor, final_mc_entropy_list,
               final_mc_entropy_list_normalize_base, evaluation_metric="q_bias", mask=None, env=None):
    # - generate indices by uniform sampling from 0 (i.e., oldest one) to agent.replay_buffer.ptr- 1, (i.e., newest one)
    indices = torch.arange(start=0,
                           end=agent.replay_buffer.size,
                           step=(agent.replay_buffer.size - 1.0) / (sample_mask_size - 1.0)
                           ).reshape((-1, 1)).to(agent.device)
    indices = torch.floor(indices).to(torch.int).cpu().numpy().reshape(-1)
    # get training samples to be evaluated.  [data batch size, info dim]
    batch = agent.replay_buffer.sample_batch(batch_size=None, idxs=indices)

    #
    if mask is None:
        masks_tensor = Tensor(batch['masks']).to(agent.device)
    else:
        masks_tensor = Tensor(mask).to(agent.device)
        sample_mask_size = mask.shape[0]

    # - evaluate scores for flip and non-flip masks.
    flip_scores = []
    non_flip_scores = []
    for i in tqdm.tqdm(range(sample_mask_size)):
        if evaluation_metric == "q_bias":
            current_masks = masks_tensor[i].repeat(eval_data_size, 1)
            flip_score, non_flip_score = _q_bias_with_flip_and_non_flip_masks(agent, obs_tensor, acts_tensor,
                                                                              current_masks, final_mc_entropy_list,
                                                                              final_mc_entropy_list_normalize_base)
        elif evaluation_metric == "reinforce":
            current_masks = masks_tensor[i].repeat(1, 1)
            flip_score, non_flip_score = _reinforce_loss_with_flip_and_non_flip_masks(agent,
                                                                                      None,
                                                                                      None,
                                                                                      current_masks,
                                                                                      None,
                                                                                      env)
        else:
            raise NotImplementedError

        flip_scores.append(flip_score)
        non_flip_scores.append(non_flip_score)
    return flip_scores, non_flip_scores, indices


def _q_bias_with_flip_and_non_flip_masks(agent, obs_tensor, acts_tensor, current_masks,
                                         final_mc_entropy_list, final_mc_entropy_list_normalize_base):
    i_q_prediction_non_flip = agent.get_ave_q_prediction_for_bias_evaluation(obs_tensor, acts_tensor,
                                                                             masks=current_masks,
                                                                             flips=False
                                                                             ).cpu().numpy().reshape(-1)
    i_q_prediction_flip = agent.get_ave_q_prediction_for_bias_evaluation(obs_tensor, acts_tensor,
                                                                         masks=current_masks,
                                                                         flips=True
                                                                         ).cpu().numpy().reshape(-1)
    # compute bias with flipped mask (i.e., param wo influence of ith data) TH20221030
    # flip_bias = np.mean((i_q_prediction_flip - final_mc_entropy_list) / final_mc_entropy_list_normalize_base)
    flip_bias = np.mean(np.abs(i_q_prediction_flip - final_mc_entropy_list) / final_mc_entropy_list_normalize_base)
    # compute bias with non-flipped mask TH20221031
    non_flip_bias = np.mean(
        np.abs(i_q_prediction_non_flip - final_mc_entropy_list) / final_mc_entropy_list_normalize_base)

    return flip_bias, non_flip_bias


def _reinforce_loss_with_flip_and_non_flip_masks(agent, obs_tensor, acts_tensor, current_masks, final_mc_entropy_list, env):

    n_eval = 10
    max_ep_len = 1000
    ep_return_list = np.zeros(n_eval)
    for j in range(n_eval):
        o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
        while not (d or (ep_len == max_ep_len)):
            # Take deterministic actions at test time
            a = agent.get_test_action(o, masks=current_masks, flips=False)
            o, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1
        ep_return_list[j] = ep_ret
    non_flip_ep_ret = np.mean(ep_return_list)

    ep_return_list = np.zeros(n_eval)
    for j in range(n_eval):
        o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
        while not (d or (ep_len == max_ep_len)):
            # Take deterministic actions at test time
            a = agent.get_test_action(o, masks=current_masks, flips=True)
            o, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1
        ep_return_list[j] = ep_ret
    flip_ep_ret = np.mean(ep_return_list)


    return flip_ep_ret, non_flip_ep_ret

def _evaluate_self_training_losses(agent, sample_mask_size):
    # - generate indices by uniform sampling from 0 (i.e., oldest one) to agent.replay_buffer.unique_id - 1, (i.e., newest one)
    indices = torch.arange(start=0,
                           end=agent.replay_buffer.size,
                           step=(agent.replay_buffer.size - 1.0) / (sample_mask_size - 1.0)
                           ).reshape((-1, 1)).to(agent.device)
    indices = torch.floor(indices).to(torch.int).cpu().numpy().reshape(-1)

    # get training samples to be evaluated.  [data batch size, info dim]
    batch = agent.replay_buffer.sample_batch(batch_size=None, idxs=indices)
    obs_tensor = Tensor(batch['obs1']).to(agent.device)
    obs_next_tensor = Tensor(batch['obs2']).to(agent.device)
    acts_tensor = Tensor(batch['acts']).to(agent.device)
    rews_tensor = Tensor(batch['rews']).unsqueeze(1).to(agent.device)
    done_tensor = Tensor(batch['done']).unsqueeze(1).to(agent.device)
    masks_tensor = Tensor(batch['masks']).to(agent.device)

    # evaluate self TD and policy losses
    # - self TD error
    non_flip_td, flip_td = _evaluate_td_with_masks(agent, obs_tensor,
                                                   acts_tensor, obs_next_tensor,
                                                   rews_tensor, done_tensor, masks_tensor)
    # - self policy loss
    non_flip_policy_loss, flip_policy_loss = _evaluate_policy_loss_with_masks(agent, obs_tensor,
                                                                              masks_tensor)

    return flip_td, non_flip_td, flip_policy_loss, non_flip_policy_loss, indices


def _evaluate_td_with_masks(agent, obs_tensor, acts_tensor, obs_next_tensor, rews_tensor, done_tensor, masks_tensor):
    # -- generate TD target (with mask)
    y_q, sample_idxs = agent.get_redq_q_target_no_grad(obs_next_tensor, rews_tensor, done_tensor,
                                                       masks_tensor=masks_tensor,
                                                       flips=False)
    # -- non-flip predictions
    q_prediction_list = []
    for q_i in range(agent.num_Q):
        q_prediction = agent.q_net_list[q_i](torch.cat([obs_tensor, acts_tensor], 1),
                                             masks=masks_tensor,
                                             flips=False)  # use ToD if flag is activate. TH20221015
        q_prediction_list.append(q_prediction)
    q_prediction_cat = torch.cat(q_prediction_list, dim=1)
    y_q = y_q.expand((-1, agent.num_Q)) if y_q.shape[1] == 1 else y_q
    non_flip_TD = torch.mean(torch.square(q_prediction_cat - y_q), dim=1).detach().cpu().numpy().reshape(-1)
    # -- flip predictions
    q_prediction_list = []
    for q_i in range(agent.num_Q):
        q_prediction = agent.q_net_list[q_i](torch.cat([obs_tensor, acts_tensor], 1),
                                             masks=masks_tensor,
                                             flips=True)  # use ToD if flag is activate. TH20221015
        q_prediction_list.append(q_prediction)
    q_prediction_cat = torch.cat(q_prediction_list, dim=1)
    flip_TD = torch.mean(torch.square(q_prediction_cat - y_q), dim=1).detach().cpu().numpy().reshape(-1)

    return non_flip_TD, flip_TD


def _evaluate_policy_loss_with_masks(agent, obs_tensor, masks_tensor):
    # -- non_flip
    a_tilda, mean_a_tilda, log_std_a_tilda, log_prob_a_tilda, _, pretanh = agent.policy_net.forward(obs_tensor,
                                                                                                    masks=masks_tensor,
                                                                                                    flips=False)
    q_a_tilda_list = []
    for sample_idx in range(agent.num_Q):
        q_a_tilda = agent.q_net_list[sample_idx](torch.cat([obs_tensor, a_tilda], 1),
                                                 masks=masks_tensor,
                                                 flips=False)  # use ToD if flag is activate. TH20221015
        q_a_tilda_list.append(q_a_tilda)
    q_a_tilda_cat = torch.cat(q_a_tilda_list, 1)
    ave_q = torch.mean(q_a_tilda_cat, dim=1, keepdim=True)
    non_flip_policy_loss = (agent.alpha * log_prob_a_tilda - ave_q).detach().cpu().numpy().reshape(-1)

    # -- flip
    a_tilda, mean_a_tilda, log_std_a_tilda, log_prob_a_tilda, _, pretanh = agent.policy_net.forward(obs_tensor,
                                                                                                    masks=masks_tensor,
                                                                                                    flips=True)
    q_a_tilda_list = []
    for sample_idx in range(agent.num_Q):
        q_a_tilda = agent.q_net_list[sample_idx](torch.cat([obs_tensor, a_tilda], 1),
                                                 masks=masks_tensor,
                                                 flips=False)  # use ToD if flag is activate. TH20221015
        q_a_tilda_list.append(q_a_tilda)
    q_a_tilda_cat = torch.cat(q_a_tilda_list, 1)
    ave_q = torch.mean(q_a_tilda_cat, dim=1, keepdim=True)
    flip_policy_loss = (agent.alpha * log_prob_a_tilda - ave_q).detach().cpu().numpy().reshape(-1)

    return non_flip_policy_loss, flip_policy_loss


# save evaluation history. TH20230420
def _save_information_list_for_influences(agent, info, info_list_name, output_dir):
    if not hasattr(agent, info_list_name):
        setattr(agent, info_list_name, [])
    info_list = getattr(agent, info_list_name)
    info_list.append(info)
    pkl = pickle.dumps(info_list)
    fout = bz2.BZ2File(output_dir + "/" + info_list_name + ".bz2", "wb", compresslevel=9)
    fout.write(pkl)
    fout.close()


def log_bias_evaluation(bias_eval_env, agent, logger, max_ep_len, alpha, gamma, n_mc_eval, n_mc_cutoff,
                        experience_cleansing=False,
                        record_training_self_training_losses=False):
    # original bias evaluation part
    final_mc_list, final_mc_entropy_list, final_obs_list, final_act_list, final_done_list = get_mc_return_with_entropy_on_reset(
        bias_eval_env, agent, max_ep_len, alpha, gamma, n_mc_eval, n_mc_cutoff)
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
    # normalized_bias_per_state = bias / final_mc_entropy_list_normalize_base
    normalized_bias_per_state = np.abs(bias) / final_mc_entropy_list_normalize_base  # use absolute norm bias 20240323
    logger.store(NormQBias=normalized_bias_per_state)
    normalized_bias_sqr_per_state = bias_squared / final_mc_entropy_list_normalize_base
    logger.store(NormQBiasSqr=normalized_bias_sqr_per_state)


    # influence estimatiuon part -------

    if not hasattr(agent, "num_epoch"):
        agent.num_epoch = -1
    agent.num_epoch += 1

    sample_mask_size = 300  # 256
    eval_data_size = bias.size

    # influence of data on over-estimation bias
    if (agent.num_epoch % 10) == 0: # evaluate per 10 epochs
        # - evaluate Q-estimation biases and reinforce loss with non-flipped / flipped masks.
        for metric in ["q_bias", "reinforce"]:
            with torch.no_grad():
                flip_score, non_flip_score, indices = _evaluateL(agent,
                                                                 sample_mask_size,
                                                                 eval_data_size,
                                                                 obs_tensor,
                                                                 acts_tensor,
                                                                 final_mc_entropy_list,
                                                                 final_mc_entropy_list_normalize_base,
                                                                 metric,
                                                                 env=bias_eval_env)
            # - record the reinforce loss and biases. TH20221030
            _save_information_list_for_influences(agent, flip_score, "list_flip_" + str(metric), logger.output_dir)
            _save_information_list_for_influences(agent, non_flip_score, "list_non_flip_" + str(metric), logger.output_dir)

            if experience_cleansing:
                with torch.no_grad():
                    print("[bias_utils.py] experience cleansing @ epoch " + str(agent.num_epoch))

                    # - evaluate the influence
                    # lower is better for flip mask (higher value is worse).
                    influence = np.array(flip_score)  # - np.array(non_flip_scores)
                    if metric == "reinforce":
                        influence = - influence

                    # cleansing
                    batch = agent.replay_buffer.sample_batch(batch_size=None, idxs=indices)
                    masks = batch['masks']
                    best_flip_mask = None
                    best_flip_score = 999999.0
                    for ind in range(masks.shape[0]):
                        if best_flip_score > influence[ind]:
                            best_flip_score = influence[ind]
                            best_flip_mask = 1.0 - masks[ind]
                    best_flip_mask = np.expand_dims(best_flip_mask, axis=0)

                    # evaluate cleansing () mask
                    _, non_flip_scores_cleansing, _ = _evaluateL(agent, sample_mask_size,
                                                                 eval_data_size, obs_tensor,
                                                                 acts_tensor, final_mc_entropy_list,
                                                                 final_mc_entropy_list_normalize_base,
                                                                 metric,
                                                                 best_flip_mask,
                                                                 env=bias_eval_env)
                    _, non_flip_scores_vanilla, _ = _evaluateL(agent, sample_mask_size,
                                                               eval_data_size, obs_tensor,
                                                               acts_tensor, final_mc_entropy_list,
                                                               final_mc_entropy_list_normalize_base,
                                                               metric,
                                                               np.ones_like(best_flip_mask) * 0.5,
                                                               env=bias_eval_env)
                    _save_information_list_for_influences(agent,
                                                          [non_flip_scores_vanilla, non_flip_scores_cleansing],
                                                          "list_" + str(metric) + "_cleansing",
                                                          logger.output_dir)

                    if metric == "q_bias":
                        valid_final_mc_entropy_list, valid_final_mc_entropy_list_normalize_base, \
                            valid_obs_tensor, valid_acts_tensor = get_mc_return_with_entropy_and_obs_act(bias_eval_env,
                                                                                                         agent,
                                                                                                         max_ep_len,
                                                                                                         alpha, gamma,
                                                                                                         n_mc_eval,
                                                                                                         n_mc_cutoff)
                        valid_eval_data_size = valid_final_mc_entropy_list.size
                        _, non_flip_scores_cleansing_valid, _ = _evaluateL(agent, sample_mask_size,
                                                                           valid_eval_data_size, valid_obs_tensor,
                                                                           valid_acts_tensor,
                                                                           valid_final_mc_entropy_list,
                                                                           valid_final_mc_entropy_list_normalize_base,
                                                                           metric,
                                                                           best_flip_mask)
                        _, non_flip_scores_vanilla_valid, _ = _evaluateL(agent, sample_mask_size,
                                                                         valid_eval_data_size, valid_obs_tensor,
                                                                         valid_acts_tensor, valid_final_mc_entropy_list,
                                                                         valid_final_mc_entropy_list_normalize_base,
                                                                         metric,
                                                                         np.ones_like(best_flip_mask) * 0.5)
                        _save_information_list_for_influences(agent,
                                                              [non_flip_scores_vanilla_valid,
                                                               non_flip_scores_cleansing_valid],
                                                              "list_" + str(metric) + "_cleansing_valid",
                                                              logger.output_dir)



    # Validation losses.
    if record_training_self_training_losses:
        # for each experience to be evaluated, evaluate TD error and policy loss for each training loss.
        with torch.no_grad():
            flip_td, non_flip_td, flip_policy_loss, non_flip_policy_loss, _ = _evaluate_self_training_losses(agent,
                                                                                                             sample_mask_size)
        # record evaluation result. TH20221030
        # - TD
        _save_information_list_for_influences(agent, flip_td, "list_flip_td", logger.output_dir)
        _save_information_list_for_influences(agent, non_flip_td, "list_non_flip_td", logger.output_dir)
        # - policy loss
        _save_information_list_for_influences(agent, flip_policy_loss, "list_flip_policy_loss", logger.output_dir)
        _save_information_list_for_influences(agent, non_flip_policy_loss, "list_non_flip_policy_loss",
                                              logger.output_dir)


