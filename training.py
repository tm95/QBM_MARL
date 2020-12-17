from datetime import datetime
import numpy as np


def train(env, agents, nb_episodes, nb_steps, logger):

    exp_time = datetime.now().strftime('%Y%m%d-%H-%M-%S')
    print("training started at {}".format(exp_time))

    for training_episode in range(nb_episodes):

        episode_rewards = np.zeros(env.nb_agents)
        state, available_actions_list = env.reset()
        step_count = 1
        all_done = False

        while step_count < nb_steps and not all_done:
            #env.render(state[0][0])
            actions_list = []

            for i in range(env.nb_agents):
                action_index = agents[i].policy(state[i], available_actions_list)
                actions_list.append(action_index)

            next_state, reward, done = env.step(actions_list, state)

            for i in range(env.nb_agents):
                agents[0].save(state[i][1], available_actions_list[actions_list[i]], next_state[i], reward[i])
                episode_rewards[i] += reward[i]

            for i in range(env.nb_agents):
                agents[0].qlearn(available_actions_list)
                
            agents[1].policy_net.Q_hh = agents[0].policy_net.Q_hh
            agents[1].policy_net.Q_vh = agents[0].policy_net.Q_vh

            step_count += 1
            state = next_state
            all_done = all(done is True for done in done)

        print("episode {} finished at step {} with and reward: {} at timestamp {}".format(
            training_episode, step_count, np.sum(episode_rewards), datetime.now().strftime('%Y%m%d-%H-%M-%S')))

        if logger is not None:
            logger.log_metric('episode_rewards', training_episode, np.sum(episode_rewards))
            logger.log_metric('episode_steps', training_episode, step_count)

            for i in range(env.nb_agents):
                logger.log_metric('episode_return_agent-{}'.format(i), training_episode, episode_rewards[i])
