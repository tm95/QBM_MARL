from datetime import datetime
import numpy as np
import time

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
            env.render(mode='human', highlight=True)
            time.sleep(0.1)

            steps += 1
            actions = np.zeros((len(env.agents), 10), dtype=int)
            obs = np.zeros((len(env.agents), 70), dtype=int)
            obs_n = np.zeros((len(env.agents), 70), dtype=int)
            action_list = []

            for i in range(len(env.agents)):
                obs[i, states[i]] = 1
                q_val = agents[i].policy(obs[i])
                action = np.argmax(q_val).item()
                action_list.append(action)
                actions[i, action] = 1

            next_state, reward, done, info = env.step(action_list)

            for agent_index in range(len(env.agents)):
                episode_rewards[agent_index] += reward[agent_index]

            # train the agent
            for i in range(len(env.agents)):
                if not done and steps != nb_steps:
                    obs_n[i, states[i]] = 1
                    q1 = (agents[i].policy(obs_n[i]))
                    agents[i].train(reward[i], q_val, q1, obs[i],
                                    np.argmax(actions[i]), np.argmax(q1))

            states = next_state
            #all_done = all(done is True for done in done)
            all_done = done

        if logger is not None:
            logger.log_metric('episode_return', training_episode, np.sum(episode_rewards))
            logger.log_metric('episode_steps', training_episode, steps)

        for i in range(len(env.agents)):
            if logger is not None:
                logger.log_metric('episode_return_agent-{}'.format(i), training_episode, episode_rewards[i])

        print("episode {} finished at step {} with reward: {} at timestamp {}".format(
            training_episode, steps, episode_rewards, datetime.now().strftime('%Y%m%d-%H-%M-%S')))

def discretize(state, grid):
    return tuple(int(np.digitize(l, g)) for l, g in zip(state, grid))