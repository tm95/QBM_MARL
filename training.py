from datetime import datetime
import numpy as np


def train(env, agent, nb_episodes, nb_steps, logger):
    exp_time = datetime.now().strftime('%Y%m%d-%H-%M-%S')

    print("training started at {}".format(exp_time))

    for training_episode in range(nb_episodes):
        steps = 0
        env.seed(seed=1234)  # Uncomment to randomize grid
        state = env.reset()
        all_done = False
        rewards = []

        # reinforcement learning loop
        while not all_done and steps < nb_steps:
            #env.render(mode='human', highlight=True)

            if training_episode > 200:
                beta = 1 * (10 / 1) ** (200 / 200)
            if training_episode <= 200:
                beta = 1 * (10 / 1) ** (training_episode / 200)

            steps += 1
            action_list = []
            action = agent.policy(state[0], beta)
            action_list.append(action)

            next_state, reward, done, info = env.step(action_list)

            reward = reward[0]

            if reward == 0:
                reward = -0.4

            agent.qlearn(state[0], action, reward)
            rewards.append(reward)
            all_done = done

        if logger is not None:
            logger.log_metric('episode_return', training_episode, np.sum(rewards))
            logger.log_metric('episode_steps', training_episode, steps)

        for i in range(len(env.agents)):
            if logger is not None:
                logger.log_metric('episode_return_agent-{}'.format(i), training_episode, np.sum(rewards))

        print("episode {} finished at step {} with reward: {} at timestamp {}".format(
            training_episode, steps, np.sum(rewards), datetime.now().strftime('%Y%m%d-%H-%M-%S')))
