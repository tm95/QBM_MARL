import training
import evaluation
from agents.rbm_agent import make_rbm_agent
from agents.dbm_agent import make_dbm_agent
from agents.qbm_agent import make_qbm_agent
import json
from dotmap import DotMap
from datetime import datetime
import os
import neptune
import sys
from gym.envs.registration import register
import gym

def train(seed, env_name):

    exp_time = datetime.now().strftime('%Y%m%d-%H-%M-%S')

    # Load action set
    params = ('params.json')
    with open(params, 'r') as f:
        params_json = json.load(f)
    params = DotMap(params_json)

    # Training or Evaluation
    for training_mode in [0]:  # 0=Training, 1=No_Training
        log_dir = os.path.join(os.getcwd(), 'experiments', 'exp-{}'.format(exp_time))

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        neptune.init('tobiasmueller/sandbox',
                     api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5tbCIsImFwa'
                               'V91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiNz'
                               'E4NDgxMmQtYTMzMC00ZTUzLTlkNDAtYWNkZTUzODExZmM4In0=')
        logger = neptune
        with neptune.create_experiment(name='sandbox', params=params_json):
            neptune.append_tag('evaluation-{}'.format(training_mode))
            run_experiment(logger, params, log_dir, training_mode, seed, env_name)


def run_experiment(logger, params, log_dir, training_mode, seed, env_name):

    if env_name == 'soccer':
        register(
            id='multigrid-soccer-v0',
            entry_point='gym_multigrid.envs:SoccerGame4HEnv10x15N2',
        )
        env = gym.make('multigrid-soccer-v0')

    else:
        register(
            id='multigrid-collect-v0',
            entry_point='gym_multigrid.envs:CollectGame4HEnv10x10N2',
        )
        env = gym.make('multigrid-collect-v0')

    weights_dir = os.path.join(log_dir, 'weights')

    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    agents = make_rbm_agent(76, 4)

    # train agent and save weights
    if training_mode == 0:
        training.train(env, agents, params.nb_episodes, params.nb_steps, logger)

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
        train(seed, 'collect')
