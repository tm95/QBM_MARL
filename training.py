from datetime import datetime
import numpy as np
import time

def train(env, agent, nb_episodes, nb_steps, logger):
    exp_time = datetime.now().strftime('%Y%m%d-%H-%M-%S')

    print("training started at {}".format(exp_time))

    for training_episode in range(nb_episodes):
        steps = 0
        env.seed(seed=1234) #Uncomment to randomize grid
        state = env.reset()
        all_done = False
        rewards = []
        discount_factor = 0.001


        # reinforcement learning loop
        while not all_done and steps < nb_steps:
            env.render(mode='human', highlight=True)
            time.sleep(0.1)

            steps += 1
            action_list = []
            #action = agent.policy(state, 10, 1)
            action = agent.policy(state)
            action_list.append(action)

            next_state, reward, done, info = env.step(action_list)
            reward = np.round(( reward * (1-(discount_factor*steps))), decimals=2)

            next_action = agent.policy(next_state, 10, 1)

            agent.qlearn(state, action, next_state, next_action, reward)

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