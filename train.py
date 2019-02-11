#!/usr/bin/env python
import argparse
import logging
from unityagents import UnityEnvironment

from r_agent import RandomAgent as Agent

log = logging.getLogger(__name__)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', default='agent1')
    parser.add_argument('-e', '--episodes', type=int)
    parser.add_argument('--no_graphics', action='store_true')
    parser.set_defaults(no_graphics=False, episodes=5)
    args = parser.parse_args()
    #
    env = UnityEnvironment(file_name="Reacher_Linux/Reacher.x86_64", no_graphics=args.no_graphics)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=False)[brain_name]
    action_size = brain.vector_action_space_size
    state = env_info.vector_observations[0]
    reward = env_info.rewards[0]
    action = 0
    state_size = len(state)
    #

    b_agent = Agent(args.model_name, state_size, action_size)
    try:
        b_agent.load()  # try to load to continue training
    except:
        pass

    for epx in range(1, args.episodes + 1):
        at_step = 0
        env_info = env.reset(train_mode=False)[brain_name]
        b_agent.reset_episode()
        while True:
            action = b_agent.act(state, use_egreedy=True)
            env_info = env.step(action)[brain_name]
            at_step += 1
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            if at_step % 100 == 0:
                log.info("ep:{} step:{} r:{} l:{}".format(epx, at_step, b_agent.cum_rewards(), b_agent.ave_loss()))
            if done:
                break
            b_agent.sense(state, action, reward, next_state, learn=True)
            state = next_state
        print("{},{}".format(epx, b_agent.cum_rewards()))
        b_agent.save()

    log.info("finished.")
