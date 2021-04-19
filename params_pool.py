from typing import Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.distributions import Categorical

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

    def __init__(self,
            input_dim: int,
            num_actions :int,
            gamma: float,
            n_step: int,
        ):

        self.policy_net = get_net(num_in=input_dim, num_out=num_actions, final_activation=nn.Softmax(dim=1))
        self.value_net  = get_net(num_in=input_dim, num_out=1,           final_activation=None)

        # learning rates are chosen heuristically
        self.policy_net_optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.value_net_optimizer  = optim.Adam(self.value_net.parameters(),  lr=1e-3)

        self.gamma = gamma
        self.n_step = n_step

        # copied from author's implementation in TF
        self.entropy_loss_weight = 0.01
        self.value_loss_weight = 0.5

    def update_networks(self, transitions: Union[NStepTransitions, MonteCarloTransitions], use_sil: bool=False) -> None:

        # these are needed by both
        dist_over_a_given_s = self.policy_net(transitions.s)
        log_p_a_given_s = torch.log(dist_over_a_given_s.gather(1, transitions.a))
        predicted_values_with_grad = self.value_net(transitions.s)
        predicted_values_without_grad = predicted_values_with_grad.detach()  # does not affect predicted_values_with_grad

        if use_sil:

            # here, we conveniently use F.relu to achieve the max(~, 0) operation
            monte_carlo_return = transitions.R
            POLICY_LOSS = - torch.mean(log_p_a_given_s * F.relu(monte_carlo_return - predicted_values_without_grad))  # equation 2
            VALUE_LOSS = torch.mean((1 / 2) * F.relu(monte_carlo_return - predicted_values_with_grad) ** 2)  # equation 3
            TOTAL_LOSS = POLICY_LOSS + self.value_loss_weight * VALUE_LOSS  # equation 1

        else:

            n_step_returns = transitions.n_step_sum_of_r + \
                             (self.gamma ** self.n_step) * self.value_net(transitions.n_step_s).detach() * (1 - transitions.done_within_n_step)
            POLICY_LOSS = - torch.mean(log_p_a_given_s * (n_step_returns - predicted_values_without_grad))  # equation 5 (first term)
            ENTROPY_LOSS = - torch.mean(torch.sum(dist_over_a_given_s * torch.log(dist_over_a_given_s), dim=1))  # equation 5 (second term)
            VALUE_LOSS = torch.mean((1 / 2) * (n_step_returns - predicted_values_with_grad) ** 2)  # equation 6
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
            dist = Categorical(self.policy_net(state))
        return int(dist.sample())



