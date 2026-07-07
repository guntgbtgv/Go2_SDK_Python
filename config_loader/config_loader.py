import yaml
import torch
import torch.nn as nn
import numpy as np
from torch.distributions.normal import Normal

# Function to load YAML configuration
def load_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

# Function to load the actor network
def load_actor_network(config, model_path='nn/model.pt'):
    model = torch.jit.load(model_path)
    model.eval()
    return model

class RunningMeanStd(nn.Module):
    def __init__(self, shape=(), epsilon=1e-08):
        super(RunningMeanStd, self).__init__()
        self.register_buffer("running_mean", torch.zeros(shape))
        self.register_buffer("running_var", torch.ones(shape))
        self.register_buffer("count", torch.ones(()))

        self.epsilon = epsilon

    def forward(self, obs, update=True):
        if update:
            self.update(obs)

        return (obs - self.running_mean) / torch.sqrt(self.running_var + self.epsilon)

    def update(self, x):
        """Updates the mean, var and count from a batch of samples."""
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, correction=0, dim=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Updates from batch mean, variance and count moments."""
        self.running_mean, self.running_var, self.count = (
            update_mean_var_count_from_moments(
                self.running_mean,
                self.running_var,
                self.count,
                batch_mean,
                batch_var,
                batch_count,
            )
        )


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, observation_space, action_apace):
        super().__init__()
        SINGLE_OBSERVATION_SPACE = observation_space
        SINGLE_ACTION_SPACE = action_apace
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(SINGLE_OBSERVATION_SPACE).prod(), 128)),
            nn.ELU(),
            layer_init(nn.Linear(128, 128)),
            nn.ELU(),
            layer_init(nn.Linear(128, 128)),
            nn.ELU(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(SINGLE_OBSERVATION_SPACE).prod(), 128)),
            nn.ELU(),
            layer_init(nn.Linear(128, 128)),
            nn.ELU(),
            layer_init(nn.Linear(128, 128)),
            nn.ELU(),
            layer_init(nn.Linear(128, np.prod(SINGLE_ACTION_SPACE)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(SINGLE_ACTION_SPACE)))

        self.obs_rms = RunningMeanStd(shape=SINGLE_OBSERVATION_SPACE)
        self.value_rms = RunningMeanStd(shape=())

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None, deterministic=False):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            if not deterministic:
                action = probs.sample()
            else:
                action = action_mean
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )
    
    def forward(self, x, deterministic=True):
        action, _, _, _ = self.get_action_and_value(self.obs_rms(x, update=False), deterministic=deterministic)
        return action