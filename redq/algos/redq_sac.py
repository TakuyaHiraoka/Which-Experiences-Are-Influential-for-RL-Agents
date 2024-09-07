from typing import Tuple, Union

import gym
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim

import redq.utils.logx
from redq.algos.core import (TanhGaussianPolicy, Mlp, soft_update_model1_with_model2, ReplayBuffer,
                             mbpo_target_entropy_dict)


class REDQSACAgent(object):
    """
    PIToD agent based on the REDQ-SAC algorithm.
    """

    def __init__(self,
                 env_name: str, obs_dim: int, act_dim: int, act_limit: int, device: torch.device,
                 hidden_sizes: Tuple[int, ...] = (256, 256),
                 replay_size: int = int(1e6),
                 batch_size: int = 256,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 polyak: float = 0.995,
                 alpha: float = 0.2,
                 auto_alpha: bool = True,
                 target_entropy: Union[str, float] = 'mbpo',
                 start_steps: int = 5000,
                 delay_update_steps: Union[str, int] = 'auto',
                 utd_ratio: int = 20,
                 num_Q: int = 10,
                 num_min: int = 2,
                 policy_update_delay: int = 20,
                 target_drop_rate: float = 0.0,
                 layer_norm: bool = False,
                 layer_norm_policy: bool = False,
                 experience_group_size: int = 5000,
                 mask_dim: int = 20,
                 ) -> None:
        """
        Initialize the PIToD agent.
        
        :param env_name: Name of the gym environment.
        :param obs_dim: Dimension of the observation space.
        :param act_dim: Dimension of the action space.
        :param act_limit: Action limit for clamping.
        :param device: CPU/GPU device to be used for computation.
        :param hidden_sizes: Sizes of the hidden layers. 
        :param replay_size: Size of the replay buffer. 
        :param batch_size: Mini-batch size. 
        :param lr: Learning rate for all networks. 
        :param gamma: Discount factor. 
        :param polyak: Hyperparameter for Polyak-averaged target networks. 
        :param alpha: SAC entropy hyperparameter. Default is 0.2.
        :param auto_alpha: Whether to use adaptive entropy. 
        :param target_entropy: Target entropy for adaptive entropy. 
        :param start_steps: Number of random data points (experiences) collected at the beginning of training. 
        :param delay_update_steps: Number of data points collected before starting updates.
        :param utd_ratio: Update-to-data ratio.
        :param num_Q: Number of Q networks in the Q ensemble. 
        :param num_min: Number of sampled Q values to take the minimum from. 
        :param policy_update_delay: Number of updates before updating the policy network. 
        :param target_drop_rate: Dropout rate for the Q-network. 
        :param layer_norm: Whether to use layer normalization for the Q-network. 
        :param layer_norm_policy: Whether to use layer normalization for the policy network.
        :param experience_group_size: size of experience group
        :param mask_dim: size of mask for turn-over dropout.
        """

        # set up networks
        self.policy_net = TanhGaussianPolicy(obs_dim=obs_dim,
                                             action_dim=act_dim,
                                             hidden_sizes=hidden_sizes,
                                             action_limit=act_limit,
                                             layer_norm=layer_norm_policy,
                                             ensemble_size=mask_dim,
                                             ).to(device)
        self.q_net_list, self.q_target_net_list = [], []
        for _ in range(num_Q):
            new_q_net = Mlp(input_size=obs_dim + act_dim,
                            output_size=1,
                            hidden_sizes=hidden_sizes,
                            target_drop_rate=target_drop_rate,
                            layer_norm=layer_norm,
                            ensemble_size=mask_dim,
                            ).to(device)
            self.q_net_list.append(new_q_net)
            new_q_target_net = Mlp(input_size=obs_dim + act_dim,
                                   output_size=1,
                                   hidden_sizes=hidden_sizes,
                                   target_drop_rate=target_drop_rate,
                                   layer_norm=layer_norm,
                                   ensemble_size=mask_dim,
                                   ).to(device)
            new_q_target_net.load_state_dict(new_q_net.state_dict())
            self.q_target_net_list.append(new_q_target_net)
        # set up optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.q_optimizer_list = [optim.Adam(q_net.parameters(), lr=lr) for q_net in self.q_net_list]
        # set up adaptive entropy (SAC adaptive)
        self.auto_alpha = auto_alpha
        if auto_alpha:
            if target_entropy == 'auto':
                self.target_entropy = - act_dim
            if target_entropy == 'mbpo':
                # target entropy for custom environments.
                mbpo_target_entropy_dict['AntTruncatedObs-v2'] = -4
                mbpo_target_entropy_dict['HumanoidTruncatedObs-v2'] = -2

                self.target_entropy = mbpo_target_entropy_dict[env_name]
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.cpu().exp().item()
        else:
            self.alpha = alpha
            self.target_entropy, self.log_alpha, self.alpha_optim = None, None, None
        # set up replay buffer
        self.replay_buffer = ReplayBuffer(obs_dim=obs_dim,
                                          act_dim=act_dim,
                                          size=replay_size,
                                          experience_group_size=experience_group_size,
                                          mask_dim=mask_dim
                                          )

        # set up other things
        self.mse_criterion = nn.MSELoss()

        # store other hyperparameters
        self.start_steps = start_steps
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.lr = lr
        self.hidden_sizes = hidden_sizes
        self.gamma = gamma
        self.polyak = polyak
        self.replay_size = replay_size
        self.alpha = alpha
        self.batch_size = batch_size
        self.num_min = num_min
        self.num_Q = num_Q
        self.utd_ratio = utd_ratio
        self.delay_update_steps = self.start_steps if delay_update_steps == 'auto' else delay_update_steps
        self.policy_update_delay = policy_update_delay
        self.device = device
        self.layer_norm = layer_norm
        self.layer_norm_policy = layer_norm_policy
        self.target_drop_rate = target_drop_rate
        self.mask_dim = mask_dim
        self.experience_group_size = experience_group_size

    def __get_current_num_data(self) -> int:
        # used to determine whether we should get action from policy or take random starting actions
        return self.replay_buffer.size

    def get_exploration_action(self, obs: np.ndarray, env: gym.Env) -> np.ndarray:
        # given an observation, output a sampled action in numpy form
        with torch.no_grad():
            if self.__get_current_num_data() > self.start_steps:
                obs_tensor = torch.Tensor(obs).unsqueeze(0).to(self.device)
                action_tensor = self.policy_net.forward(obs_tensor,
                                                        deterministic=False,
                                                        return_log_prob=False)[0]
                action = action_tensor.cpu().numpy().reshape(-1)
            else:
                action = env.action_space.sample()
        return action

    def get_test_action(self,
                        obs: np.ndarray,
                        masks: Union[torch.Tensor, None] = None,
                        flips: bool = False) -> np.ndarray:
        # given an observation, output a deterministic action in numpy form
        with torch.no_grad():
            obs_tensor = torch.Tensor(obs).unsqueeze(0).to(self.device)
            action_tensor = self.policy_net.forward(obs_tensor,
                                                    deterministic=True,
                                                    return_log_prob=False,
                                                    masks=masks,
                                                    flips=flips)[0]
            action = action_tensor.cpu().numpy().reshape(-1)
        return action

    def get_action_and_logprob_for_bias_evaluation(self, obs: np.ndarray) -> Tuple[np.ndarray, torch.Tensor]:
        # given an observation, output a sampled action in numpy form
        with torch.no_grad():
            obs_tensor = torch.Tensor(obs).unsqueeze(0).to(self.device)
            action_tensor, _, _, log_prob_a_tilda, _, _, = self.policy_net.forward(obs_tensor,
                                                                                   deterministic=False,
                                                                                   return_log_prob=True)
            action = action_tensor.cpu().numpy().reshape(-1)
        return action, log_prob_a_tilda

    def get_ave_q_prediction_for_bias_evaluation(self,
                                                 obs_tensor: torch.Tensor,
                                                 acts_tensor: torch.Tensor,
                                                 masks: Union[torch.Tensor, None] = None,
                                                 flips: bool = False
                                                 ) -> torch.Tensor:
        # given obs_tensor and act_tensor, output Q prediction
        q_prediction_list = []
        for q_i in range(self.num_Q):
            q_prediction = self.q_net_list[q_i](torch.cat([obs_tensor, acts_tensor], 1),
                                                masks=masks,
                                                flips=flips)
            q_prediction_list.append(q_prediction)
        q_prediction_cat = torch.cat(q_prediction_list, dim=1)
        average_q_prediction = torch.mean(q_prediction_cat, dim=1)
        return average_q_prediction

    def store_data(self, o, a, r, o2, d):
        # store one transition to the buffer
        self.replay_buffer.store(o, a, r, o2, d)

    def sample_data(self, batch_size: int) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # sample data from replay buffer
        batch = self.replay_buffer.sample_batch(batch_size)
        obs_tensor = Tensor(batch['obs1']).to(self.device)
        obs_next_tensor = Tensor(batch['obs2']).to(self.device)
        acts_tensor = Tensor(batch['acts']).to(self.device)
        rews_tensor = Tensor(batch['rews']).unsqueeze(1).to(self.device)
        done_tensor = Tensor(batch['done']).unsqueeze(1).to(self.device)
        masks_tensor = Tensor(batch['masks']).to(self.device)
        return obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor, masks_tensor

    def get_redq_q_target_no_grad(self,
                                  obs_next_tensor: torch.Tensor,
                                  rews_tensor: torch.Tensor,
                                  done_tensor: torch.Tensor,
                                  masks_tensor: Union[torch.Tensor, None] = None,
                                  flips: bool = False) -> torch.Tensor:
        # compute SAC target
        sample_idxs = np.random.choice(self.num_Q, self.num_min, replace=False)
        with torch.no_grad():
            # Q target is min of a subset of Q values
            a_tilda_next, _, _, log_prob_a_tilda_next, _, _ = self.policy_net.forward(obs_next_tensor,
                                                                                      masks=masks_tensor,
                                                                                      flips=flips
                                                                                      )
            q_prediction_next_list = []
            for sample_idx in sample_idxs:
                q_prediction_next = self.q_target_net_list[sample_idx](torch.cat([obs_next_tensor, a_tilda_next], 1),
                                                                       masks=masks_tensor,
                                                                       flips=flips
                                                                       )
                q_prediction_next_list.append(q_prediction_next)
            q_prediction_next_cat = torch.cat(q_prediction_next_list, 1)
            min_q, min_indices = torch.min(q_prediction_next_cat, dim=1, keepdim=True)
            next_q_with_log_prob = min_q - self.alpha * log_prob_a_tilda_next
            y_q = rews_tensor + self.gamma * (1 - done_tensor) * next_q_with_log_prob

        return y_q

    def train(self, logger: redq.utils.logx.EpochLogger) -> None:
        # this function is called after each datapoint is collected.
        # when we have very limited data, we don't make updates
        num_update = 0 if self.__get_current_num_data() <= self.delay_update_steps else self.utd_ratio

        for i_update in range(num_update):
            obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor, masks_tensor = self.sample_data(
                self.batch_size)

            """Q loss"""
            y_q = self.get_redq_q_target_no_grad(obs_next_tensor, rews_tensor, done_tensor,
                                                 masks_tensor=masks_tensor,
                                                 flips=False)
            q_prediction_list = []
            for q_i in range(self.num_Q):
                q_prediction = self.q_net_list[q_i](torch.cat([obs_tensor, acts_tensor], 1),
                                                    masks=masks_tensor,
                                                    flips=False)
                q_prediction_list.append(q_prediction)
            q_prediction_cat = torch.cat(q_prediction_list, dim=1)
            y_q = y_q.expand((-1, self.num_Q)) if y_q.shape[1] == 1 else y_q
            q_loss_all = self.mse_criterion(q_prediction_cat, y_q) * self.num_Q

            for q_i in range(self.num_Q):
                self.q_optimizer_list[q_i].zero_grad()
            q_loss_all.backward()

            """policy and alpha loss"""
            if ((i_update + 1) % self.policy_update_delay == 0) or (i_update == num_update - 1):
                # get policy loss
                a_tilda, mean_a_tilda, log_std_a_tilda, log_prob_a_tilda, _, pretanh = self.policy_net.forward(
                    obs_tensor,
                    masks=masks_tensor,
                    flips=False)
                q_a_tilda_list = []
                for sample_idx in range(self.num_Q):
                    self.q_net_list[sample_idx].requires_grad_(False)
                    q_a_tilda = self.q_net_list[sample_idx](torch.cat([obs_tensor, a_tilda], 1),
                                                            masks=masks_tensor,
                                                            flips=False)
                    q_a_tilda_list.append(q_a_tilda)
                q_a_tilda_cat = torch.cat(q_a_tilda_list, 1)
                ave_q = torch.mean(q_a_tilda_cat,
                                   dim=1,
                                   keepdim=True)
                policy_loss = (self.alpha * log_prob_a_tilda - ave_q).mean()
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                for sample_idx in range(self.num_Q):
                    self.q_net_list[sample_idx].requires_grad_(True)

                # get alpha loss
                if self.auto_alpha:
                    alpha_loss = -(self.log_alpha * (log_prob_a_tilda + self.target_entropy).detach()).mean()
                    self.alpha_optim.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optim.step()
                    self.alpha = self.log_alpha.cpu().exp().item()
                else:
                    alpha_loss = Tensor([0])

            """update networks"""
            for q_i in range(self.num_Q):
                self.q_optimizer_list[q_i].step()

            if ((i_update + 1) % self.policy_update_delay == 0) or (i_update == num_update - 1):
                self.policy_optimizer.step()

            # polyak averaged Q target networks
            for q_i in range(self.num_Q):
                soft_update_model1_with_model2(self.q_target_net_list[q_i], self.q_net_list[q_i], self.polyak)

            # by default, only log for the last update out of <num_update> updates
            if i_update == num_update - 1:
                logger.store(LossPi=policy_loss.cpu().item(), LossQ1=q_loss_all.cpu().item() / self.num_Q,
                             LossAlpha=alpha_loss.cpu().item(), Q1Vals=q_prediction.detach().cpu().numpy(),
                             Alpha=self.alpha, LogPi=log_prob_a_tilda.detach().cpu().numpy(),
                             PreTanh=pretanh.abs().detach().cpu().numpy().reshape(-1),
                             )

        # if there is no update, store 0 to prevent logging problems.
        if num_update == 0:
            logger.store(LossPi=0, LossQ1=0, LossAlpha=0, Q1Vals=0, Alpha=0, LogPi=0, PreTanh=0)

    # reset parameters of Q and policy networks.
    def reset(self) -> None:
        # reset all components as with original reset paper. https://arxiv.org/abs/2205.07802
        # set up networks
        self.policy_net = TanhGaussianPolicy(obs_dim=self.obs_dim,
                                             action_dim=self.act_dim,
                                             hidden_sizes=self.hidden_sizes,
                                             action_limit=self.act_limit,
                                             layer_norm=self.layer_norm_policy,
                                             ensemble_size=self.mask_dim,
                                             ).to(self.device)
        self.q_net_list, self.q_target_net_list = [], []
        for _ in range(self.num_Q):
            new_q_net = Mlp(input_size=self.obs_dim + self.act_dim,
                            output_size=1,
                            hidden_sizes=self.hidden_sizes,
                            target_drop_rate=self.target_drop_rate,
                            layer_norm=self.layer_norm,
                            ensemble_size=self.mask_dim,
                            ).to(self.device)
            self.q_net_list.append(new_q_net)
            new_q_target_net = Mlp(input_size=self.obs_dim + self.act_dim,
                                   output_size=1,
                                   hidden_sizes=self.hidden_sizes,
                                   target_drop_rate=self.target_drop_rate,
                                   layer_norm=self.layer_norm,
                                   ensemble_size=self.mask_dim,
                                   ).to(self.device)
            new_q_target_net.load_state_dict(new_q_net.state_dict())
            self.q_target_net_list.append(new_q_target_net)
        # set up optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.q_optimizer_list = [optim.Adam(q_net.parameters(), lr=self.lr) for q_net in self.q_net_list]

        # set up adaptive entropy (SAC adaptive)
        if self.auto_alpha:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=self.lr)
            self.alpha = self.log_alpha.cpu().exp().item()
        else:
            self.target_entropy, self.log_alpha, self.alpha_optim = None, None, None
        # set up other things
        self.mse_criterion = nn.MSELoss()
