import numpy as np
import torch
from collections import namedtuple
import random

# very exhaustive but as a result code is very easy to read ;)

Transition = namedtuple('Transition', 's a r done')
Transitions = namedtuple('Transitions', 's a r done')  # functionally the same as Transition
NStepTransitions = namedtuple('Batch', 's a n_step_sum_of_r n_step_s done_within_n_step')  # each item corresponds to a tensor

class SequentialBuffer:

    def __init__(self, gamma: float, n_step: int):
        self.memory = []
        self.gamma = gamma
        self.n_step = n_step

    def push(self, transition: Transition) -> None:
        self.memory.append(transition)

    def instantiate_NStepTransitions_and_empty_buffer(self) -> NStepTransitions:

        """We empty the buffer because this is an on-policy algorithm."""

        dummy_transition = Transition(np.zeros((self.memory[0].s.shape)), 0, 0, 0)

        memory_with_dummies = self.memory + [dummy_transition] * (self.n_step - 1)

        transitions = Transitions(*zip(*memory_with_dummies))
        length = len(self.memory)  # not computed with memory_with_dummies

        s        = torch.tensor(transitions.s[:-(self.n_step-1) or None],  dtype=torch.float).view(length, -1)
        a        = torch.tensor(transitions.a[:-(self.n_step-1) or None],  dtype=torch.long ).view(length,  1)
        n_step_s = torch.tensor(transitions.s[self.n_step-1:], dtype=torch.float).view(length, -1)
        # last few are dummies and their values will be ignored conveniently through done_within_n_step

        n_step_sum_of_r = np.zeros((len(self.memory),))
        done_within_n_step = np.zeros((len(self.memory),))  # including the zeroth and the (n-1)th step

        for t in range(len(transitions.r[:-(self.n_step-1) or None])):  # t = 0, 1, ..., T-1, (T, ..., T+n_step-1), where bracket terms due to dummies
            sum = 0
            for i in range(self.n_step):  # from 0 to n_step - 1 inclusive; exactly what we want
                sum += (self.gamma ** i) * transitions.r[t+i]
                if transitions.done[t+i]:
                    done_within_n_step[t] = 1  # indicates that we shouldn't care about the value of n_step_s
                    break
            n_step_sum_of_r[t] = sum

        n_step_sum_of_r = torch.tensor(n_step_sum_of_r, dtype=torch.float).view(length, 1)
        done_within_n_step = torch.tensor(done_within_n_step, dtype=torch.long).view(length, 1)

        self.memory = []

        return NStepTransitions(s, a, n_step_sum_of_r, n_step_s, done_within_n_step)

# for SIL
TransitionWithoutDone = namedtuple('TransitionWithoutDone', 's a r')
TransitionWithoutDoneWithReturn = namedtuple('TransitionWithoutDoneWithReturn', 's a R')
MonteCarloTransitions = namedtuple('MonteCarloTransitions', 's a R')

class SILBuffer:

    def __init__(self, gamma: float):
        self.memory = []
        self.current_episode = []
        self.gamma = gamma

    def push(self, transition: TransitionWithoutDone) -> None:
        self.current_episode.append(transition)

    def process_and_empty_current_episode(self) -> None:
        discounted_return = 0
        for transition in reversed(self.current_episode):
            discounted_return = transition.r + self.gamma * discounted_return
            self.memory.append(TransitionWithoutDoneWithReturn(transition.s, transition.a, discounted_return))
        self.current_episode = []

    def ready_for(self, batch_size: int) -> bool:
        return len(self.memory) >= batch_size

    def sample(self, batch_size: int) -> MonteCarloTransitions:
        transitions = random.sample(self.memory, batch_size)
        transitions = MonteCarloTransitions(*zip(*transitions))
        s = torch.tensor(transitions.s, dtype=torch.float).view(batch_size, -1)
        a = torch.tensor(transitions.a, dtype=torch.long ).view(batch_size,  1)
        R = torch.tensor(transitions.R, dtype=torch.float).view(batch_size,  1)
        return MonteCarloTransitions(s, a, R)