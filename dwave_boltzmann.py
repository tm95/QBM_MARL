from agent_utils import make_test_agent
from env_utils import make_env
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import neptune
import json
from dotmap import DotMap

def run(env, agent, logger):

    exp_time = datetime.now().strftime('%Y%m%d-%H-%M-%S')
    print("training started at {}".format(exp_time))

    # Initialize constants
    nb_episodes = 500
    nb_steps = 300
    step_count_list = list()
    fidelity_list = list()

    for training_episode in range(nb_episodes):
        rewards = []

        state, available_actions_list = env.reset()

        step_count = 1
        fidelity_count = 1
        done = False

        while step_count < nb_steps and not done:
            # env.render(agent_state_tuple[0])

            action_index = agent.policy(state, available_actions_list)
            next_state, fidelity, reward, done = env.step(action_index, state, state[0])

            agent.save(state[1], available_actions_list[action_index], next_state, reward)

            agent.qlearn(available_actions_list)

            rewards.append(reward)
            fidelity_count += fidelity
            step_count += 1

            state = next_state


        fidelity_list.append(fidelity_count/step_count)
        step_count_list.append(step_count)

        print("episode {} finished at step {} with fidelity: {} and reward: {} at timestamp {}".format(
            training_episode, step_count, np.round(fidelity_count/step_count, decimals=2), np.sum(rewards), datetime.now().strftime('%Y%m%d-%H-%M-%S')))

        if logger is not None:
            logger.log_metric('episode_fidelity', training_episode, np.round(fidelity_count/step_count, decimals=2))
            logger.log_metric('episode_rewards', training_episode, np.sum(rewards))
            logger.log_metric('episode_steps', training_episode, step_count)


def print_results(step_count_list, fidelity_list):
    plt.plot(range(len(step_count_list)), step_count_list, 'b-')
    plt.plot(range(len(step_count_list)), step_count_list, 'ro')
    # plt.show()
    plt.savefig('agent_step_history.png')

    plt.clf()

    plt.plot(range(len(fidelity_list)), fidelity_list, 'b-')
    plt.plot(range(len(fidelity_list)), fidelity_list, 'ro')
    plt.show()


if __name__ == '__main__':
    for lr in [0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001]:
        env = make_env()

        observation_space = env.observation_space()
        action_space = env.action_space()

        agent = make_test_agent(observation_space, action_space, lr)

        # Load action set
        params = ('params.json')
        with open(params, 'r') as f:
            params_json = json.load(f)
        params = DotMap(params_json)

        neptune.init('tobiasmueller/sandbox',
                    api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5tbCIsImFwa'
                            'V91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiNz'
                            'E4NDgxMmQtYTMzMC00ZTUzLTlkNDAtYWNkZTUzODExZmM4In0=')
        logger = neptune
        with neptune.create_experiment(name='sandbox', params=params_json):
            neptune.append_tag('lr-{}'.format(lr))
            run(env, agent, logger)
