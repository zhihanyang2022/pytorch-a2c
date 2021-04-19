from replay_buffer import Transition, SequentialBuffer
import time

buf = SequentialBuffer(gamma=0.95, n_step=3)
for i in range(10):
    buf.push(Transition([1], [0], 1, 0))
    buf.push(Transition([2], [0], 1, 0))
    buf.push(Transition([3], [0], 1, 0))
    buf.push(Transition([4], [0], 1, 1))

start = time.perf_counter()

n_step_transitions = buf.instantiate_NStepTransitions_and_empty_buffer()

print('==== states')
print(n_step_transitions.s.squeeze())
print('==== actions')
print(n_step_transitions.a.squeeze())
print('==== n_step_sum_of_r')
print(n_step_transitions.n_step_sum_of_r.squeeze())
print('==== n_step_s')
print(n_step_transitions.n_step_s.squeeze())
print('==== done_within_n_step')
print(n_step_transitions.done_within_n_step.squeeze())

print(f'Time taken is {time.perf_counter() - start} seconds.')