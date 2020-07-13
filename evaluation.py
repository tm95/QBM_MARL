from datetime import datetime
import numpy as np


def evaluate(env, agents, nb_episodes, nb_steps, logger):
    exp_time = datetime.now().strftime('%Y%m%d-%H-%M-%S')

    print("evaluation started at {}".format(exp_time))

    for evaluation_episode in range(nb_episodes):
        steps = 0
        states = env.reset()
        all_done = False
        episode_rewards = np.zeros(env.n, dtype=np.float32)

        # classic reinforcement learning loop
        while not all_done and steps < nb_steps:
            steps += 1
            actions = np.zeros((env.n, 10), dtype=int)
            action_list = []

            for i in range(env.n):
                action = agents[i].policy(states[i])
                action_list.append(action)
                actions[i, action] = 1

                next_state, reward, done, info = env.step(actions)

            for agent_index in range(env.n):
                agents[agent_index].save(states[agent_index],
                                         actions[agent_index],
                                         next_state[agent_index],
                                         reward[agent_index],
                                         done[agent_index])
                episode_rewards[agent_index] += reward[agent_index]

            states = next_state
            all_done = all(done is True for done in done)

        if logger is not None:
            logger.log_metric('episode_return', evaluation_episode, np.sum(episode_rewards))
            logger.log_metric('episode_steps', evaluation_episode, steps)

        for i in range(env.n):
            logger.log_metric('episode_return_agent-{}'.format(i), evaluation_episode, episode_rewards[i])

        print("evaluation {} finished at step {} with reward: {} at timestamp {}".format(
            evaluation_episode, steps, episode_rewards, exp_time))
