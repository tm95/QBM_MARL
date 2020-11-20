from datetime import datetime
import numpy as np
import random

def train(env, agent, nb_episodes, nb_steps, logger):
    exp_time = datetime.now().strftime('%Y%m%d-%H-%M-%S')

    print("training started at {}".format(exp_time))

    for training_episode in range(nb_episodes):
        steps = 0
        env.seed(seed=1234)  # Uncomment to randomize grid
        state = env.reset()
        all_done = False
        rewards = []
        beta = 1

        # reinforcement learning loop
        while not all_done and steps < nb_steps:
            #env.render(mode='human', highlight=True)

            steps += 1
            action_list = []
            #action, q, hh = agent.policy(state[0], beta)
            action = agent.policy(state[0], beta)
            action_list.append(action)

            next_state, reward, done, info = env.step(action_list)

            if reward == 0:
                reward = -1.0

            #print (reward)

            #agent.qlearn(state[0], action, reward, next_state[0], 0.01, q, hh)
            agent.qlearn(state[0], action, reward)
            rewards.append(reward)
            state = next_state
            all_done = done

        if logger is not None:
            logger.log_metric('episode_return', training_episode, np.sum(rewards))
            logger.log_metric('episode_steps', training_episode, steps)

        for i in range(len(env.agents)):
            if logger is not None:
                logger.log_metric('episode_return_agent-{}'.format(i), training_episode, np.sum(rewards))

        print("episode {} finished at step {} with reward: {} at timestamp {}".format(
            training_episode, steps, np.sum(rewards), datetime.now().strftime('%Y%m%d-%H-%M-%S')))
