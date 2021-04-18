import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.distributions import Categorical
from typing import Union
import torch.nn.functional as F

from replay_buffer import NStepTransitions, MonteCarloTransitions

def get_net(
        num_in:int,
        num_out:int,
        final_activation,  # e.g. nn.Tanh
        num_hidden_layers:int=1,
        num_neurons_per_hidden_layer:int=64
    ) -> nn.Sequential:

    layers = []

    layers.extend([
        nn.Linear(num_in, num_neurons_per_hidden_layer),
        nn.ReLU(),
    ])

    for _ in range(num_hidden_layers):
        layers.extend([
            nn.Linear(num_neurons_per_hidden_layer, num_neurons_per_hidden_layer),
            nn.ReLU(),
        ])

    layers.append(nn.Linear(num_neurons_per_hidden_layer, num_out))

    if final_activation is not None:
        layers.append(final_activation)

    return nn.Sequential(*layers)

class ParamsPool:

    """
    Based on paper actor critic with self-imitation learning

    """

    def __init__(self,
            input_dim: int,
            num_actions :int,
            gamma: float,
            n_step: int,
        ):

        self.policy_net = get_net(num_in=input_dim, num_out=num_actions, final_activation=nn.Softmax(dim=1))
        self.value_net  = get_net(num_in=input_dim, num_out=1,           final_activation=None)

        self.policy_net_optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.value_net_optimizer  = optim.Adam(self.value_net.parameters(),  lr=1e-3)

        # for optimization

        self.gamma = gamma
        self.n_step = n_step

        self.entropy_loss_weight = 0.01
        self.value_loss_weight = 0.5

    def update_networks(self, transitions: Union[NStepTransitions, MonteCarloTransitions], use_sil: bool=False) -> None:

        dist_over_a_given_s = self.policy_net(transitions.s)
        log_p_a_given_s = torch.log(dist_over_a_given_s.gather(1, transitions.a))

        if use_sil:
            n_step_returns = transitions.R
        else:
            n_step_returns = transitions.n_step_sum_of_r + \
                             (self.gamma ** self.n_step) * self.value_net(transitions.n_step_s).detach() * (1 - transitions.done_within_n_step)

        predicted_values = self.value_net(transitions.s).detach()

        if use_sil:  # we conveniently use relu to do the max(~,0) operation
            POLICY_LOSS = - torch.mean(log_p_a_given_s * F.relu(n_step_returns - predicted_values))  # equation 2
        else:
            POLICY_LOSS = - torch.mean(log_p_a_given_s * (n_step_returns - predicted_values))  # equation 5 (first term)

        if not use_sil:
            ENTROPY_LOSS = - torch.mean(torch.sum(dist_over_a_given_s * torch.log(dist_over_a_given_s), dim=1))  # equation 5 (second term)

        predicted_values = self.value_net(transitions.s)  # TODO: try optimize this step
        if use_sil:
            VALUE_LOSS = torch.mean((1 / 2) * F.relu(n_step_returns - predicted_values) ** 2)  # equation 3
        else:
            VALUE_LOSS = torch.mean((1 / 2) * (n_step_returns - predicted_values) ** 2)  # equation 6

        if use_sil:
            TOTAL_LOSS = POLICY_LOSS \
                         + self.value_loss_weight * VALUE_LOSS # equation 1
        else:
            TOTAL_LOSS = POLICY_LOSS \
                         + self.entropy_loss_weight * ENTROPY_LOSS \
                         + self.value_loss_weight * VALUE_LOSS  # equation 4

        self.policy_net_optimizer.zero_grad()
        self.value_net_optimizer.zero_grad()

        TOTAL_LOSS.backward()

        # doing a gradient clipping between -1 and 1 is equivalent to using Huber loss
        # guaranteed to improve stability so no harm in using at all
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        for param in self.value_net.parameters():
            param.grad.data.clamp_(-1, 1)

        self.policy_net_optimizer.step()
        self.value_net_optimizer.step()

    def act(self, state: np.array) -> int:
        state = torch.tensor(state).unsqueeze(0).float()
        with torch.no_grad():
            try:
                dist = Categorical(self.policy_net(state))
            except:
                print(state)

                print(self.policy_net(state))
        return int(dist.sample())



