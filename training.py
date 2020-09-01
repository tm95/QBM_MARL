from datetime import datetime
import numpy as np
import time

def train(env, agent, nb_episodes, nb_steps, logger):
    exp_time = datetime.now().strftime('%Y%m%d-%H-%M-%S')

    print("training started at {}".format(exp_time))

    for training_episode in range(nb_episodes):
        steps = 0
        state = env.reset()
        a1 = np.random.randint(7, size=1)
        all_done = False
        rewards = []

        # reinforcement learning loop
        while not all_done and steps < nb_steps:
            env.render(mode='human', highlight=True)
            time.sleep(0.1)

            steps += 1
            actions = np.zeros(7, dtype=int)
            action_list = []

            #print (len(state))

            #print (a1)

            #q_val = agent.policy(obs)
            #action = np.argmax(q_val).item()
            #actions[action] = 1
            action_list.append(a1)

            next_state, reward, done, info = env.step(action_list)
            print (reward)

            a2 = agent.policy(next_state)
            q2 = agent.calculate_free_energy(next_state)
            #actions[action] = 1
            #action_list.append(action)

            rewards.append(reward)

            q1 = agent.calculate_free_energy(state)

            #agent.train(reward[0], q1, q2, state, a1, a2)

            state = next_state
            a1 = a2
            all_done = done

        if logger is not None:
            logger.log_metric('episode_return', training_episode, np.sum(rewards))
            logger.log_metric('episode_steps', training_episode, steps)

        for i in range(len(env.agents)):
            if logger is not None:
                logger.log_metric('episode_return_agent-{}'.format(i), training_episode, np.sum(rewards))

        print("episode {} finished at step {} with reward: {} at timestamp {}".format(
            training_episode, steps, np.sum(rewards), datetime.now().strftime('%Y%m%d-%H-%M-%S')))

def discretize(state, grid):
    return tuple(int(np.digitize(l, g)) for l, g in zip(state, grid))