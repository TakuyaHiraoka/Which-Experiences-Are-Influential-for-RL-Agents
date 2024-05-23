import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Distribution, Normal

# following SAC authors' and OpenAI implementation
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
ACTION_BOUND_EPSILON = 1E-6
# these numbers are from the MBPO paper
mbpo_target_entropy_dict = {'Hopper-v2': -1, 'HalfCheetah-v2': -3, 'Walker2d-v2': -3, 'Ant-v2': -4, 'Humanoid-v2': -2}
mbpo_epoches = {'Hopper-v2': 125, 'Walker2d-v2': 300, 'Ant-v2': 300, 'HalfCheetah-v2': 400, 'Humanoid-v2': 300}

# dropout ensembles. # TH20240318 TODO specify via argument.
NUM_TURN_OVER_ENSEMBLES = 20

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer
    """

    def __init__(self, obs_dim, act_dim, size, hidden_sizes, size_of_mask_cluster=5000):
        """
        :param obs_dim: size of observation
        :param act_dim: size of the action
        :param size: size of the buffer
        """
        # init buffers as numpy arrays
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

        # mask. TH20230410
        self.hidden_size = hidden_sizes[-1]
        self.size_of_mask_cluster = size_of_mask_cluster
        self.masks_buf = np.zeros([size, NUM_TURN_OVER_ENSEMBLES], dtype=np.int32)

        #
        self.obs_dim = obs_dim
        self.act_dim = act_dim

    def store(self, obs, act, rew, next_obs, done):
        """
        data will get stored in the pointer's location
        """
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done

        # added TH 202230410 <-  modified TH 20240314
        if (self.ptr % self.size_of_mask_cluster) == 0:
            # i.e., mask rate = 0.5
            self.ids_zero_elem = np.random.permutation(NUM_TURN_OVER_ENSEMBLES)[:int(NUM_TURN_OVER_ENSEMBLES / 2)]
        mask = np.ones([NUM_TURN_OVER_ENSEMBLES], dtype=np.int32)
        mask[self.ids_zero_elem] = 0.0
        self.masks_buf[self.ptr] = mask

        assert not ((self.ptr + 1) % self.max_size) == 0, ("Experiences for PI+ToD paper does not allow cyclic "
                                                           "pointer.")

        # move the pointer to store in next location in buffer
        self.ptr = (self.ptr + 1) % self.max_size
        # keep track of the current buffer size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32, idxs=None):
        """
        :param batch_size: size of minibatch
        :param idxs: specify indexes if you want specific data points
        :return: mini-batch data as a dictionary
        """
        if idxs is None:
            idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs],
                    masks=self.masks_buf[idxs],  # added TH20230410
                    idxs=idxs)

    # initialize buffer, leaving data specified by index. TH20230208
    def clean_buffer(self, index_of_data_to_leave):
        # get data to leave
        index_of_data_to_leave = np.sort(index_of_data_to_leave)  # sort old to new one to keep order consistent
        index_of_data_to_leave = index_of_data_to_leave.astype(np.int)
        obs1 = self.obs1_buf[index_of_data_to_leave]
        obs2 = self.obs2_buf[index_of_data_to_leave]
        acts = self.acts_buf[index_of_data_to_leave]
        rews = self.rews_buf[index_of_data_to_leave]
        done = self.done_buf[index_of_data_to_leave]
        masks = self.masks_buf[index_of_data_to_leave]  # 20230410
        # init buffer and attbute (ptr and size). leave unique_id as is.
        self.obs1_buf = np.zeros([self.max_size, self.obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([self.max_size, self.obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([self.max_size, self.act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(self.max_size, dtype=np.float32)
        self.done_buf = np.zeros(self.max_size, dtype=np.float32)
        # self.masks_buf = np.zeros([self.max_size, self.hidden_size], dtype=np.float32)
        self.masks_buf = np.zeros([self.max_size, NUM_TURN_OVER_ENSEMBLES], dtype=np.int32)
        # set left data to new buffer. set ptr and size.
        self.obs1_buf[:obs1.shape[0]] = obs1
        self.obs2_buf[:obs2.shape[0]] = obs2
        self.acts_buf[:acts.shape[0]] = acts
        self.rews_buf[:rews.shape[0]] = rews
        self.done_buf[:done.shape[0]] = done
        self.masks_buf[:masks.shape[0]] = masks
        self.ptr, self.size = index_of_data_to_leave.shape[0], index_of_data_to_leave.shape[0]


class Mlp(nn.Module):
    def __init__(
            self,
            input_size,
            output_size,
            hidden_sizes,
            hidden_activation=F.relu,
            #
            target_drop_rate=0.0,
            layer_norm=False,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.hidden_sizes = hidden_sizes
        self.target_drop_rate = target_drop_rate
        self.apply_layer_norm = layer_norm

        hidden_input_size = input_size
        self.hidden_weights = []
        for i in range(0, len(hidden_sizes)):  # TODO make function to create (init) weight and bias from hidden size
            param = nn.Parameter(torch.empty(NUM_TURN_OVER_ENSEMBLES, hidden_input_size, hidden_sizes[i]))
            torch.nn.init.xavier_uniform_(param, gain=1)
            self.hidden_weights.append(param)
            hidden_input_size = hidden_sizes[i]
        self.hidden_weights = torch.nn.ParameterList(self.hidden_weights)
        self.output_weight = nn.Parameter(torch.empty(NUM_TURN_OVER_ENSEMBLES, hidden_input_size, output_size))
        torch.nn.init.xavier_uniform_(self.output_weight, gain=1)
        # bias
        self.hidden_biases = []
        for ind in range(0, len(hidden_sizes)):
            param = nn.Parameter(torch.empty(NUM_TURN_OVER_ENSEMBLES, 1, hidden_sizes[ind]))
            torch.nn.init.xavier_uniform_(param, gain=1)
            self.hidden_biases.append(param)
        self.hidden_biases = torch.nn.ParameterList(self.hidden_biases)
        self.output_bias = nn.Parameter(torch.empty(NUM_TURN_OVER_ENSEMBLES, 1, output_size))
        torch.nn.init.constant_(self.output_bias, 0)
        # Layer Normalization, Dropout, etc.
        if self.apply_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_sizes[0])
        if self.target_drop_rate > 0.0:
            self.dropout = nn.Dropout(self.target_drop_rate)

    def forward(self, input, masks=None, flips=False):
        hidden = input.unsqueeze(0).expand(NUM_TURN_OVER_ENSEMBLES, -1, -1)  # [n_mlps, batch_size, input_size]

        # forward
        for hidden_weight, hidden_bias in zip(self.hidden_weights, self.hidden_biases):
            hidden = torch.bmm(hidden, hidden_weight) + hidden_bias  # [n_mlps, batch_size, hidden_size]
            hidden = hidden.view(-1, hidden.size(-1))  # [n_mlps * batch_size, hidden_size]
            if self.target_drop_rate > 0.0:
                hidden = self.dropout(hidden)
            if self.apply_layer_norm:
                hidden = self.layer_norm(hidden)
            hidden = self.hidden_activation(hidden)
            hidden = hidden.view(NUM_TURN_OVER_ENSEMBLES, -1, hidden.size(-1))  # [n_mlps, batch_size, hidden_size]
        output = torch.bmm(hidden, self.output_weight) + self.output_bias  # [n_mlps, batch_size, output_size]

        # masking
        if masks is not None:
            masks_t = torch.transpose(masks, 0, 1).unsqueeze(-1)  # [batch_size, masks_size] -> [mask_size, batch_size, 1]
        else:
            if not hasattr(self, "_default_mask"):
                self._default_mask = torch.ones(NUM_TURN_OVER_ENSEMBLES, 1, 1).to(input.device)  # [mask_size, 1, 1]
            masks_t = self._default_mask.expand(NUM_TURN_OVER_ENSEMBLES, input.shape[0], 1)  # ->  [mask_size, batch_size, 1]
        if  (masks is not None) and flips:
            masks_t = (1.0 - masks_t)

        # weighted sum
        masked_output = torch.sum(output * masks_t, dim=0) / torch.sum(masks_t, dim=0)

        return masked_output


class TanhNormal(Distribution):
    """
    Represent distribution of X where
        X ~ tanh(Z)
        Z ~ N(mean, std)
    Note: this is not very numerically stable.
    """

    def __init__(self, normal_mean, normal_std, epsilon=1e-6):
        """
        :param normal_mean: Mean of the normal distribution
        :param normal_std: Std of the normal distribution
        :param epsilon: Numerical stability epsilon when computing log-prob.
        """
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = Normal(normal_mean, normal_std)
        self.epsilon = epsilon

    def log_prob(self, value, pre_tanh_value=None):
        """
        return the log probability of a value
        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        # use arctanh formula to compute arctanh(value)
        if pre_tanh_value is None:
            pre_tanh_value = torch.log(
                (1 + value) / (1 - value)
            ) / 2
        return self.normal.log_prob(pre_tanh_value) - \
            torch.log(1 - value * value + self.epsilon)

    def sample(self, return_pretanh_value=False):
        """
        Gradients will and should *not* pass through this operation.
        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        z = self.normal.sample().detach()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def rsample(self, return_pretanh_value=False):
        """
        Sampling in the reparameterization case.
        Implement: tanh(mu + sigma * eksee)
        with eksee~N(0,1)
        z here is mu+sigma+eksee
        """
        z = (
                self.normal_mean +
                self.normal_std *
                Normal(  # this part is eksee~N(0,1)
                    torch.zeros(self.normal_mean.size()),
                    torch.ones(self.normal_std.size())
                ).sample()
        )
        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)


class TanhGaussianPolicy(Mlp):
    """
    A Gaussian policy network with Tanh to enforce action limits
    """

    def __init__(
            self,
            obs_dim,
            action_dim,
            hidden_sizes,
            hidden_activation=F.relu,
            action_limit=1.0,
            layer_norm=False
    ):
        super().__init__(
            input_size=obs_dim,
            output_size=action_dim,
            hidden_sizes=hidden_sizes,
            hidden_activation=hidden_activation,
            layer_norm=layer_norm
        )
        last_hidden_size = obs_dim
        if len(hidden_sizes) > 0:
            last_hidden_size = hidden_sizes[-1]
        # this is the layer that gives log_std, init this layer with small weight and bias
        self.log_std_weight = nn.Parameter(torch.empty(NUM_TURN_OVER_ENSEMBLES,  # TODO use init function
                                                       last_hidden_size,
                                                       action_dim))
        torch.nn.init.xavier_uniform_(self.log_std_weight, gain=1)
        self.log_std_bias = nn.Parameter(torch.empty(NUM_TURN_OVER_ENSEMBLES,
                                                     1,
                                                     action_dim))
        torch.nn.init.constant_(self.log_std_bias, 0)

        # action limit: for example, humanoid has an action limit of -0.4 to 0.4
        self.action_limit = action_limit
        # self.apply(weights_init_)

    def forward(self, obs, deterministic=False, return_log_prob=True, masks=None, flips=False):
        """
        :param obs: Observation
        :param deterministic: If True, take deterministic (test) action
        :param return_log_prob: If True, return a sample and its log probability
        """
        hidden = obs.unsqueeze(0).expand(NUM_TURN_OVER_ENSEMBLES, -1, -1)  # [n_mlps, batch_size, input_size]

        # forward
        for hidden_weight, hidden_bias in zip(self.hidden_weights, self.hidden_biases):
            hidden = torch.bmm(hidden, hidden_weight) + hidden_bias  # [n_mlps, batch_size, hidden_size]
            hidden = hidden.view(-1, hidden.size(-1))  # [n_mlps * batch_size, hidden_size]
            if self.target_drop_rate > 0.0:
                hidden = self.dropout(hidden)
            if self.apply_layer_norm:
                hidden = self.layer_norm(hidden)
            hidden = self.hidden_activation(hidden)
            hidden = hidden.view(NUM_TURN_OVER_ENSEMBLES, -1, hidden.size(-1))  # [n_mlps, batch_size, hidden_size]
        output = torch.bmm(hidden, self.output_weight) + self.output_bias  # [n_mlps, batch_size, output_size]
        log_std = torch.bmm(hidden, self.log_std_weight) + self.log_std_bias

        # masking
        if masks is not None:
            masks_t = torch.transpose(masks, 0, 1).unsqueeze(-1)  # [batch_size, masks_size] -> [mask_size, batch_size, 1]
        else:
            if not hasattr(self, "_default_mask"):
                self._default_mask = torch.ones(NUM_TURN_OVER_ENSEMBLES, 1, 1).to(obs.device)  # [mask_size, 1, 1]
            masks_t = self._default_mask.expand(NUM_TURN_OVER_ENSEMBLES, obs.shape[0], 1)  # [mask_size, batch_size, 1]
        if (masks is not None) and flips:
            masks_t = (1.0 - masks_t)

        # weighted sum
        masked_output = torch.sum(output * masks_t, dim=0) / torch.sum(masks_t, dim=0)
        masked_log_std = torch.sum(log_std * masks_t, dim=0) / torch.sum(masks_t, dim=0)

        mean = masked_output
        log_std = masked_log_std
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(log_std)
        normal = Normal(mean, std)

        if deterministic:
            pre_tanh_value = mean
            action = torch.tanh(mean)
        else:
            pre_tanh_value = normal.rsample()
            action = torch.tanh(pre_tanh_value)

        if return_log_prob:
            log_prob = normal.log_prob(pre_tanh_value)
            log_prob -= torch.log(1 - action.pow(2) + ACTION_BOUND_EPSILON)
            log_prob = log_prob.sum(1, keepdim=True)
        else:
            log_prob = None

        return (
            action * self.action_limit, mean, log_std, log_prob, std, pre_tanh_value,
        )


def soft_update_model1_with_model2(model1, model2, rou):
    """
    used to polyak update a target network
    :param model1: a pytorch model
    :param model2: a pytorch model of the same class
    :param rou: the update is model1 <- rou*model1 + (1-rou)model2
    """
    for model1_param, model2_param in zip(model1.parameters(), model2.parameters()):
        model1_param.data.copy_(rou * model1_param.data + (1 - rou) * model2_param.data)


def test_agent(agent, test_env, max_ep_len, logger, n_eval=1):
    """
    This will test the agent's performance by running <n_eval> episodes
    During the runs, the agent should only take deterministic action
    This function assumes the agent has a <get_test_action()> function
    :param agent: agent instance
    :param test_env: the environment used for testing
    :param max_ep_len: max length of an episode
    :param logger: logger to store info in
    :param n_eval: number of episodes to run the agent
    :return: test return for each episode as a numpy array
    """
    ep_return_list = np.zeros(n_eval)
    for j in range(n_eval):
        o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
        while not (d or (ep_len == max_ep_len)):
            # Take deterministic actions at test time
            a = agent.get_test_action(o)
            o, r, d, _ = test_env.step(a)
            ep_ret += r
            ep_len += 1
        ep_return_list[j] = ep_ret
        if logger is not None:
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
    return ep_return_list
