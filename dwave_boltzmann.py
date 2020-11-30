from agent_utils import make_test_agent
from env_utils import make_env
from datetime import datetime
import numpy as np
import neptune
import json
from dotmap import DotMap

def run(env, agents, logger):

    exp_time = datetime.now().strftime('%Y%m%d-%H-%M-%S')
    print("training started at {}".format(exp_time))

    # Initialize constants
    nb_episodes = 500
    nb_steps = 300

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
                agents[i].save(state[i][1], available_actions_list[actions_list[i]], next_state[i], reward[i])
                episode_rewards[i] += reward[i]

            for i in range(env.nb_agents):
                agents[i].qlearn(available_actions_list)

            #if training_episode > 5:
             #   agents[1].policy_net.Q_hh = agents[0].policy_net.Q_hh
              #  agents[1].policy_net.Q_vh = agents[0].policy_net.Q_vh

            step_count += 1
            state = next_state
            all_done = any(done is True for done in done)

        print("episode {} finished at step {} with and reward: {} at timestamp {}".format(
            training_episode, step_count, np.sum(episode_rewards), datetime.now().strftime('%Y%m%d-%H-%M-%S')))

        if logger is not None:
            logger.log_metric('episode_rewards', training_episode, np.sum(episode_rewards))
            logger.log_metric('episode_steps', training_episode, step_count)

            for i in range(env.nb_agents):
                logger.log_metric('episode_return_agent-{}'.format(i), training_episode, episode_rewards[i])


if __name__ == '__main__':
    for lr in [0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001]:
        nb_agents = 2

        env = make_env(nb_agents)

        observation_space = env.observation_space()
        action_space = env.action_space()
        agents = []
        for i in range(nb_agents):
            agents.append(make_test_agent(observation_space, action_space, lr))

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
            run(env, agents, logger)
