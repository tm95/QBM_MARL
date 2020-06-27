from datetime import datetime
import numpy as np


def train(env, agents, nb_episodes, nb_steps, logger):
    exp_time = datetime.now().strftime('%Y%m%d-%H-%M-%S')

    print("training started at {}".format(exp_time))

    for training_episode in range(nb_episodes):
        steps = 0
        states = env.reset()
        all_done = False
        episode_rewards = np.zeros(len(agents), dtype=np.float32)

        # reinforcement learning loop
        while not all_done and steps < nb_steps:
            env.render()
            steps += 1
            actions = np.zeros((env.n, 10), dtype=int)
            action_list = []

            for i in range(env.n):
                action = agents[i].policy(states[i])
                action_list.append(action)
                actions[i, action] = 1

            next_state, reward, done, info = env.step(actions)
            print (reward)

            for agent_index in range(env.n):
                agents[agent_index].save(states[agent_index],
                                         action_list[agent_index],
                                         next_state[agent_index],
                                         reward[agent_index],
                                         done[agent_index])
                episode_rewards[agent_index] += reward[agent_index]

            # train the agent
            for i in range(env.n):
                if not done[i] and steps != nb_steps:
                    agents[i].train()

            states = next_state
            all_done = all(done is True for done in done)

        if logger is not None:
            logger.log_metric('episode_return', training_episode, np.sum(episode_rewards))
            logger.log_metric('episode_steps', training_episode, steps)

        for i in range(env.n):
            if logger is not None:
                logger.log_metric('episode_return_agent-{}'.format(i), training_episode, episode_rewards[i])

        print("episode {} finished at step {} with reward: {} at timestamp {}".format(
            training_episode, steps, episode_rewards, datetime.now().strftime('%Y%m%d-%H-%M-%S')))
