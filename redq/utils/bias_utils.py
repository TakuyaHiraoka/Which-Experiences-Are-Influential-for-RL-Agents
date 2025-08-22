from typing import Dict, Tuple, Union, Any, List

from gym import Env
from redq.algos.redq_sac import REDQSACAgent
from redq.utils.logx import Logger

import math

import numpy as np
import torch
from torch import Tensor

import pickle
import bz2

import tqdm
import imageio

import os

def get_mc_return_with_entropy_on_reset(bias_eval_env: Env,
                                        agent: REDQSACAgent,
                                        max_ep_len: int,
                                        alpha: float,
                                        gamma: float,
                                        n_mc_eval: int,
                                        n_mc_cutoff: int) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    final_mc_list = np.zeros(0)
    final_mc_entropy_list = np.zeros(0)
    final_obs_list = []
    final_act_list = []
    final_done_list = []

    while final_mc_list.shape[0] < n_mc_eval:
        # we continue if agent haven't collected enough data
        o = bias_eval_env.reset()
        # temporary lists
        reward_list, log_prob_a_tilda_list, obs_list, act_list, done_list = [], [], [], [], []

        r, d, ep_ret, ep_len = 0, False, 0, 0
        for i_step in range(max_ep_len):  # run an episode
            with torch.no_grad():
                a, log_prob_a_tilda = agent.get_action_and_logprob_for_bias_evaluation(o)
            obs_list.append(o)
            act_list.append(a)
            o, r, d, _ = bias_eval_env.step(a)
            done_list.append(d)
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


def get_mc_return_with_entropy_and_obs_act(bias_eval_env: Env,
                                           agent: REDQSACAgent,
                                           max_ep_len: int,
                                           alpha: float,
                                           gamma: float,
                                           n_mc_eval: int,
                                           n_mc_cutoff: int) \
        -> Tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
    _, final_mc_entropy_list, final_obs_list, final_act_list, final_done_list = get_mc_return_with_entropy_on_reset(
        bias_eval_env, agent, max_ep_len, alpha, gamma, n_mc_eval, n_mc_cutoff)
    obs_tensor = Tensor(final_obs_list).to(agent.device)
    acts_tensor = Tensor(final_act_list).to(agent.device)
    final_mc_entropy_list_normalize_base = final_mc_entropy_list.copy()
    final_mc_entropy_list_normalize_base = np.abs(final_mc_entropy_list_normalize_base)
    final_mc_entropy_list_normalize_base[final_mc_entropy_list_normalize_base < 10] = 10

    return final_mc_entropy_list, final_mc_entropy_list_normalize_base, obs_tensor, acts_tensor

def _evaluate_performance_with_masks(agent: REDQSACAgent,
                                     sample_mask_size: int,
                                     eval_data_size: int,
                                     obs_tensor: torch.Tensor,
                                     acts_tensor: torch.Tensor,
                                     final_mc_entropy_list: np.ndarray,
                                     final_mc_entropy_list_normalize_base: np.ndarray,
                                     evaluation_metric: str = "q_bias",
                                     mask: Union[np.ndarray, None] = None,
                                     env: Union[Env, None] = None,
                                     n_eval: int = 10,
                                     video_dir: Union[str, None] = None) \
        -> Tuple[List[np.float64],List[np.float64],np.ndarray]:
    #  Generate indices using uniform sampling from the start of the replay buffer (oldest experience)
    #  to the latest experience, defined by `agent.replay_buffer.size - 1.
    indices = torch.arange(start=0,
                           end=agent.replay_buffer.size,
                           step=(agent.replay_buffer.size - 1.0) / (sample_mask_size - 1.0)
                           ).reshape((-1, 1)).to(agent.device)
    indices = torch.floor(indices).to(torch.int).cpu().numpy().reshape(-1)
    # get training samples to be evaluated. [data batch size, info dim]
    batch = agent.replay_buffer.sample_batch(batch_size=None, idxs=indices)

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
        elif evaluation_metric == "return":
            current_masks = masks_tensor[i].repeat(1, 1)
            flip_score, non_flip_score = _return_with_flip_and_non_flip_masks(agent, current_masks, env,
                                                                              n_eval, video_dir)
        else:
            raise NotImplementedError

        flip_scores.append(flip_score)
        non_flip_scores.append(non_flip_score)
    return flip_scores, non_flip_scores, indices


def _q_bias_with_flip_and_non_flip_masks(agent: REDQSACAgent,
                                         obs_tensor: torch.Tensor,
                                         acts_tensor: torch.Tensor,
                                         current_masks: torch.Tensor,
                                         final_mc_entropy_list: np.ndarray,
                                         final_mc_entropy_list_normalize_base: np.ndarray) \
        -> Tuple[np.float64, np.float64]:
    i_q_prediction_non_flip = agent.get_ave_q_prediction_for_bias_evaluation(obs_tensor, acts_tensor,
                                                                             masks=current_masks,
                                                                             flips=False
                                                                             ).cpu().numpy().reshape(-1)
    i_q_prediction_flip = agent.get_ave_q_prediction_for_bias_evaluation(obs_tensor, acts_tensor,
                                                                         masks=current_masks,
                                                                         flips=True
                                                                         ).cpu().numpy().reshape(-1)
    # compute bias with the flipped mask
    flip_bias = np.mean(np.abs(i_q_prediction_flip - final_mc_entropy_list) / final_mc_entropy_list_normalize_base)
    # compute bias with the non-flipped mask
    non_flip_bias = np.mean(
        np.abs(i_q_prediction_non_flip - final_mc_entropy_list) / final_mc_entropy_list_normalize_base)

    return flip_bias, non_flip_bias


def _return_with_flip_and_non_flip_masks(agent: REDQSACAgent, current_masks: torch.Tensor, env: Env,
                                         n_eval: int = 10, video_dir: Union[str, None] = None) \
        -> Tuple[np.float64, np.float64]:
    max_ep_len = 1000

    def _return(agent: REDQSACAgent, current_masks: torch.Tensor, env: Env,
                flips: bool, video_dir: Union[str, None] = None) -> np.float64:
        image_list = []

        ep_return_list = np.zeros(n_eval)
        for j in range(n_eval):
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            while not (d or (ep_len == max_ep_len)):
                # record image frame
                if video_dir is not None:
                    image = env.render(mode="rgb_array", height=512, width=512)
                    image_list.append(image)

                # Take deterministic actions at test time
                a = agent.get_test_action(o, masks=current_masks, flips=flips)
                o, r, d, _ = env.step(a)
                ep_ret += r
                ep_len += 1
            ep_return_list[j] = ep_ret

        # make video from the recorded frames.
        if video_dir is not None:
            if flips:
                video_name = "flip"
            else:
                video_name = "non_flip"
            imageio.mimsave(video_dir + video_name + ".mp4", image_list, fps=int(1.0 / 0.002 / 7.0), format="mp4")
        return np.mean(ep_return_list)

    # evaluate return with non-flipped mask.
    non_flip_ep_ret = _return(agent, current_masks, env, flips=False, video_dir=video_dir)
    # evaluate return with non-flipped mask.
    flip_ep_ret = _return(agent, current_masks, env, flips=True, video_dir=video_dir)

    return flip_ep_ret, non_flip_ep_ret


def _evaluate_self_training_losses(agent: REDQSACAgent, sample_mask_size: int) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # - generate indices uniformly from 0 (i.e., oldest one) to agent.replay_buffer.unique_id - 1 (i.e., newest one)
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
    non_flip_td, flip_td = _evaluate_td_with_masks(agent, obs_tensor, acts_tensor, obs_next_tensor,
                                                   rews_tensor, done_tensor, masks_tensor)
    non_flip_policy_loss, flip_policy_loss = _evaluate_policy_loss_with_masks(agent, obs_tensor, masks_tensor)
    return flip_td, non_flip_td, flip_policy_loss, non_flip_policy_loss, indices


def _evaluate_td_with_masks(agent: REDQSACAgent, obs_tensor: torch.Tensor, acts_tensor: torch.Tensor,
                            obs_next_tensor: torch.Tensor, rews_tensor: torch.Tensor, done_tensor: torch.Tensor,
                            masks_tensor: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    # -- generate TD target with mask
    y_q = agent.get_redq_q_target_no_grad(obs_next_tensor, rews_tensor, done_tensor,
                                          masks_tensor=masks_tensor,
                                          flips=False)
    # -- non-flip predictions
    q_prediction_list = []
    for q_i in range(agent.num_Q):
        q_prediction = agent.q_net_list[q_i](torch.cat([obs_tensor, acts_tensor], 1),
                                             masks=masks_tensor,
                                             flips=False)
        q_prediction_list.append(q_prediction)
    q_prediction_cat = torch.cat(q_prediction_list, dim=1)
    y_q = y_q.expand((-1, agent.num_Q)) if y_q.shape[1] == 1 else y_q
    non_flip_td = torch.mean(torch.square(q_prediction_cat - y_q), dim=1).detach().cpu().numpy().reshape(-1)
    # -- flip predictions
    q_prediction_list = []
    for q_i in range(agent.num_Q):
        q_prediction = agent.q_net_list[q_i](torch.cat([obs_tensor, acts_tensor], 1),
                                             masks=masks_tensor,
                                             flips=True)
        q_prediction_list.append(q_prediction)
    q_prediction_cat = torch.cat(q_prediction_list, dim=1)
    flip_td = torch.mean(torch.square(q_prediction_cat - y_q), dim=1).detach().cpu().numpy().reshape(-1)

    return non_flip_td, flip_td


def _evaluate_policy_loss_with_masks(agent: REDQSACAgent, obs_tensor: torch.Tensor, masks_tensor: torch.Tensor) \
        -> Tuple[np.ndarray, np.ndarray]:
    # -- non_flip
    a_tilda, mean_a_tilda, log_std_a_tilda, log_prob_a_tilda, _, pretanh = agent.policy_net.forward(obs_tensor,
                                                                                                    masks=masks_tensor,
                                                                                                    flips=False)
    q_a_tilda_list = []
    for sample_idx in range(agent.num_Q):
        q_a_tilda = agent.q_net_list[sample_idx](torch.cat([obs_tensor, a_tilda], 1),
                                                 masks=masks_tensor,
                                                 flips=False)
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
                                                 flips=False)
        q_a_tilda_list.append(q_a_tilda)
    q_a_tilda_cat = torch.cat(q_a_tilda_list, 1)
    ave_q = torch.mean(q_a_tilda_cat, dim=1, keepdim=True)
    flip_policy_loss = (agent.alpha * log_prob_a_tilda - ave_q).detach().cpu().numpy().reshape(-1)

    return non_flip_policy_loss, flip_policy_loss


# save evaluation result.
def _save_information_list_for_influences(agent: REDQSACAgent,
                                          info: Any,
                                          info_list_name: str,
                                          output_dir: str) -> None:
    # Retrieve the info list. Initialize if the agent does not have the list.
    if not hasattr(agent, info_list_name):
        setattr(agent, info_list_name, [])
    info_list = getattr(agent, info_list_name)
    info_list.append(info)

    pkl = pickle.dumps(info_list)
    output_file = os.path.join(output_dir, info_list_name + ".bz2")
    with bz2.BZ2File(output_file, "wb", compresslevel=9) as fout:
        fout.write(pkl)


def log_evaluation(bias_eval_env: Env,
                   agent: REDQSACAgent,
                   logger: Logger,
                   max_ep_len: int,
                   alpha: float,
                   gamma: float,
                   n_mc_eval: int,
                   n_mc_cutoff: int,
                   experience_cleansing: bool = False,
                   dump_trajectory_for_demo: bool = False,
                   record_training_self_training_losses: bool = False,
                   influence_estimation_interval: int = 10,
                   n_eval: int = 10
                   ) -> None:
    # bias evaluation part
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
    normalized_bias_per_state = np.abs(bias) / final_mc_entropy_list_normalize_base
    logger.store(NormQBias=normalized_bias_per_state)
    normalized_bias_sqr_per_state = bias_squared / final_mc_entropy_list_normalize_base
    logger.store(NormQBiasSqr=normalized_bias_sqr_per_state)

    # influence estimation for return part and Q-estimation bias and policy/Q-function amendment part
    if not hasattr(agent, "num_epoch"):
        agent.num_epoch = -1
    agent.num_epoch += 1

    sample_mask_size = np.ceil(float(agent.replay_buffer.max_size) / agent.replay_buffer.experience_group_size).astype(int)
    eval_data_size = bias.size

    if (agent.num_epoch % influence_estimation_interval) == 0:
        # - evaluate Q-estimation biases and return with non-flipped / flipped masks.
        for metric in ["q_bias", "return"]:
            with torch.no_grad():
                flip_score, non_flip_score, indices = _evaluate_performance_with_masks(agent,
                                                                                       sample_mask_size,
                                                                                       eval_data_size,
                                                                                       obs_tensor,
                                                                                       acts_tensor,
                                                                                       final_mc_entropy_list,
                                                                                       final_mc_entropy_list_normalize_base,
                                                                                       metric,
                                                                                       env=bias_eval_env,
                                                                                       n_eval=n_eval)
            # - record return and biases.
            _save_information_list_for_influences(agent, flip_score, "list_flip_" + str(metric), logger.output_dir)
            _save_information_list_for_influences(agent, non_flip_score, "list_non_flip_" + str(metric),
                                                  logger.output_dir)

            if experience_cleansing:
                with torch.no_grad():
                    # - evaluate the influence
                    # lower is better for flip mask (higher value is worse).
                    influence = np.array(flip_score)  # - np.array(non_flip_scores)
                    if metric == "return":
                        influence = - influence

                    # amendment find best flipped mask w_*
                    batch = agent.replay_buffer.sample_batch(batch_size=None, idxs=indices)
                    masks = batch['masks']
                    best_flip_mask = None
                    best_flip_score = 999999.0
                    best_ind = -1
                    for ind in range(masks.shape[0]):
                        if best_flip_score > influence[ind]:
                            best_flip_score = influence[ind]
                            best_flip_mask = 1.0 - masks[ind]
                            best_ind = indices[ind]
                    best_flip_mask = np.expand_dims(best_flip_mask, axis=0)

                    # dumping most negatively/positively influential trajectory. #TODO remove or clean up
                    if dump_trajectory_for_demo and metric == "return":
                        experience_group_size = agent.replay_buffer.experience_group_size

                        delete_start = math.floor(best_ind / float(experience_group_size)) * experience_group_size
                        delete_end = delete_start + experience_group_size
                        delete_index = np.arange(delete_start, delete_end)
                        # dump trajectory deleted at amendment.
                        batch_deleted = agent.replay_buffer.sample_batch(batch_size=None, idxs=delete_index)
                        _save_information_list_for_influences(agent, batch_deleted, "list_trajectory_deleted",
                                                              logger.output_dir)
                        # dump most positively influential experiences.
                        worst_flip_score = -999999.0
                        worst_ind = -1
                        for ind in range(masks.shape[0]):
                            if worst_flip_score < influence[ind]:
                                worst_flip_score = influence[ind]
                                worst_ind = indices[ind]
                        delete_start = math.floor(worst_ind / float(experience_group_size)) * experience_group_size
                        delete_end = delete_start + experience_group_size
                        delete_index = np.arange(delete_start, delete_end)
                        # dump trajectory deleted at amendment.
                        batch_left = agent.replay_buffer.sample_batch(batch_size=None, idxs=delete_index)
                        _save_information_list_for_influences(agent, batch_left, "list_trajectory_left",
                                                              logger.output_dir)

                    # evaluate cleansing () mask
                    _, non_flip_scores_cleansing, _ = _evaluate_performance_with_masks(agent,
                                                                                       sample_mask_size,
                                                                                       eval_data_size,
                                                                                       obs_tensor,
                                                                                       acts_tensor,
                                                                                       final_mc_entropy_list,
                                                                                       final_mc_entropy_list_normalize_base,
                                                                                       metric,
                                                                                       best_flip_mask,
                                                                                       env=bias_eval_env,
                                                                                       n_eval=n_eval,
                                                                                       video_dir=logger.output_dir + "/amended_at_" + str(agent.num_epoch) + "_" if dump_trajectory_for_demo else None
                                                                                       )
                    _, non_flip_scores_vanilla, _ = _evaluate_performance_with_masks(agent,
                                                                                     sample_mask_size,
                                                                                     eval_data_size,
                                                                                     obs_tensor,
                                                                                     acts_tensor,
                                                                                     final_mc_entropy_list,
                                                                                     final_mc_entropy_list_normalize_base,
                                                                                     metric,
                                                                                     np.ones_like(best_flip_mask) * 0.5,
                                                                                     env=bias_eval_env,
                                                                                     n_eval=n_eval,
                                                                                     video_dir=logger.output_dir + "/vanilla_at_" + str(agent.num_epoch) + "_" if dump_trajectory_for_demo else None
                                                                                     )
                    _save_information_list_for_influences(agent,
                                                          [non_flip_scores_vanilla, non_flip_scores_cleansing],
                                                          "list_" + str(metric) + "_cleansing",
                                                          logger.output_dir)

                    if metric == "q_bias":  # TODO remove?
                        valid_final_mc_entropy_list, valid_final_mc_entropy_list_normalize_base, \
                            valid_obs_tensor, valid_acts_tensor = get_mc_return_with_entropy_and_obs_act(bias_eval_env,
                                                                                                         agent,
                                                                                                         max_ep_len,
                                                                                                         alpha,
                                                                                                         gamma,
                                                                                                         n_mc_eval,
                                                                                                         n_mc_cutoff)
                        valid_eval_data_size = valid_final_mc_entropy_list.size
                        _, non_flip_scores_cleansing_valid, _ = _evaluate_performance_with_masks(agent,
                                                                                                 sample_mask_size,
                                                                                                 valid_eval_data_size,
                                                                                                 valid_obs_tensor,
                                                                                                 valid_acts_tensor,
                                                                                                 valid_final_mc_entropy_list,
                                                                                                 valid_final_mc_entropy_list_normalize_base,
                                                                                                 metric,
                                                                                                 best_flip_mask)
                        _, non_flip_scores_vanilla_valid, _ = _evaluate_performance_with_masks(agent,
                                                                                               sample_mask_size,
                                                                                               valid_eval_data_size,
                                                                                               valid_obs_tensor,
                                                                                               valid_acts_tensor,
                                                                                               valid_final_mc_entropy_list,
                                                                                               valid_final_mc_entropy_list_normalize_base,
                                                                                               metric,
                                                                                               np.ones_like(best_flip_mask) * 0.5)
                        _save_information_list_for_influences(agent,
                                                              [non_flip_scores_vanilla_valid, non_flip_scores_cleansing_valid],
                                                              "list_" + str(metric) + "_cleansing_valid",
                                                              logger.output_dir)

    # evaluate self-influence.
    if record_training_self_training_losses:
        # for each experience to be evaluated, evaluate TD error and policy loss for each training loss.
        with torch.no_grad():
            flip_td, non_flip_td, flip_policy_loss, non_flip_policy_loss, _ = _evaluate_self_training_losses(agent,
                                                                                                             sample_mask_size)
        # record evaluation result.
        # - TD
        _save_information_list_for_influences(agent,
                                              flip_td,
                                              "list_flip_td",
                                              logger.output_dir)
        _save_information_list_for_influences(agent,
                                              non_flip_td,
                                              "list_non_flip_td",
                                              logger.output_dir)
        # - policy loss
        _save_information_list_for_influences(agent,
                                              flip_policy_loss,
                                              "list_flip_policy_loss",
                                              logger.output_dir)
        _save_information_list_for_influences(agent,
                                              non_flip_policy_loss,
                                              "list_non_flip_policy_loss",
                                              logger.output_dir)
