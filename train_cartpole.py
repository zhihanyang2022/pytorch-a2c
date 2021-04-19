import argparse
import gym
from replay_buffer import SequentialBuffer, Transition, SILBuffer, TransitionWithoutDone
from params_pool import ParamsPool
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--n_step', type=int)
parser.add_argument('--run_id', type=int)
parser.add_argument('--verbose', type=int)
parser.add_argument('--delay_reward', type=int)  # a cumulative reward is given every 40 steps
parser.add_argument('--use_sil', type=int)
args = parser.parse_args()

env = gym.make('CartPole-v0')
buf = SequentialBuffer(gamma=0.95, n_step=args.n_step)
if args.use_sil:
    buf2 = SILBuffer(gamma=0.95)
param = ParamsPool(
    input_dim=env.observation_space.shape[0],
    num_actions=env.action_space.n,
    gamma=0.95,
    n_step=args.n_step
)

num_batches = 1000
num_episodes_per_batch = 20
num_episodes = num_batches * num_episodes_per_batch

# logging
if args.delay_reward:
    if args.use_sil:
        group = f'n_step={args.n_step} delay_reward sil'
    else:
        group = f'n_step={args.n_step} delay_reward'
else:
    group = f'n_step={args.n_step}'

print('Group:', group)

wandb.init(
    project='a2c',
    entity='yangz2',
    group=group,
    settings=wandb.Settings(_disable_stats=True),
    name=f'run_id={args.run_id}'
)

for e in range(num_episodes):

    obs = env.reset()

    total_reward = 0

    if args.delay_reward:
        total_steps = 0
        delayed_reward = 0

    while True:

        # ==================================================
        # getting the tuple (s, a, r, s', done)
        # ==================================================

        action = param.act(obs)
        next_obs, reward, done, _ = env.step(action)

        if args.delay_reward:
            total_steps += 1
            delayed_reward += reward
            if (total_steps % 40 == 0):
                reward = delayed_reward
                delayed_reward = 0
            else:
                reward = 0

        # no need to keep track of max time-steps, because the environment
        # is wrapped with TimeLimit automatically (timeout after 1000 steps)

        total_reward += reward

        # ==================================================
        # storing it to the buffer
        # ==================================================

        buf.push(Transition(obs, action, reward, done))
        if args.use_sil:
            buf2.push(TransitionWithoutDone(obs, action, reward))

        # ==================================================
        # check done
        # ==================================================

        if done: break

        obs = next_obs

    # ==================================================
    # after each episode
    # ==================================================

    wandb.log({"return": total_reward})

    if args.use_sil:
        buf2.process_and_empty_current_episode()

    if e % num_episodes_per_batch == 0:
        batch_index = e / num_episodes_per_batch
        param.update_networks(buf.instantiate_NStepTransitions_and_empty_buffer())
        if args.use_sil:
            if buf2.ready_for(batch_size=64):
                for i in range(5):
                    param.update_networks(buf2.sample(batch_size=64), use_sil=True)
        if args.verbose:
            print(f'Batch {batch_index:4.0f} | Return for last episode {total_reward:7.3f}')

env.close()