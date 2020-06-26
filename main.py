import training
import evaluation
from agents.dqn_agent import make_dqn_agent
from envs.make_env import make_env
import json
from dotmap import DotMap
from datetime import datetime
import os
#import neptune
import sys


def train(seed):

    exp_time = datetime.now().strftime('%Y%m%d-%H-%M-%S')

    # Load action set
    params = ('params.json')
    with open(params, 'r') as f:
        params_json = json.load(f)
    params = DotMap(params_json)

    # Training or Evaluation
    for training_mode in [0, 1]:  # 0=Training, 1=No_Training
        log_dir = os.path.join(os.getcwd(), 'experiments', 'exp-{}'.format(exp_time))

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        #neptune.init('tobiasmueller/test',
        #             api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5tbCIsImFwa'
        #                       'V91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiNz'
        #                       'E4NDgxMmQtYTMzMC00ZTUzLTlkNDAtYWNkZTUzODExZmM4In0=')
        #logger = neptune
        logger = None
        #with neptune.create_experiment(name='test', params=params_json):
        #    neptune.append_tag('evaluation-{}'.format(training_mode))
        run_experiment(logger, params, log_dir, training_mode, seed)


def run_experiment(logger, params, log_dir, training_mode, seed):

    agents = []
    env = make_env('simple_adversary')
    observation_shape = list(env.observation_space)
    number_of_actions = env.action_space
    print (observation_shape)

    weights_dir = os.path.join(log_dir, 'weights')

    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    # make agents and load weights
    for i in range(env.n):
        agent = make_dqn_agent(params, observation_shape[i].shape, number_of_actions[i].n, seed)
        print (observation_shape[i].shape)
        agents.append(agent)

    # train agent and save weights
    if training_mode == 0:
        training.train(env, agents, params.nb_episodes, params.nb_steps, logger)

        for i in range(env.n):
            agents[i].save_weights(os.path.join(weights_dir, "weights-{}.pth".format(i)))

    # evaluate agents and load weights
    if training_mode == 1:
        for i, agent in enumerate(agents):
            agent.epsilon = params.epsilon_min
            agent.load_weights(os.path.join(log_dir, 'weights', "weights-{}.pth".format(i)))
        evaluation.evaluate(env, agents, params.nb_evaluation_episodes, params.nb_steps, logger)


if __name__ == '__main__':
    args = sys.argv
    seed = 1589174148213878
    for run in range(10):
        train(seed)
