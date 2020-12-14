from env.gridworld import make_env
from agents.rbm_agent import make_rbm_agent
from agents.qbm_agent import make_qbm_agent
import training
import json
from dotmap import DotMap
import neptune
import sys


def train():

    # Load parameters
    params = ('params.json')
    with open(params, 'r') as f:
        params_json = json.load(f)
    params = DotMap(params_json)

    neptune.init('tobiasmueller/qmarl',
                 api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5tbCIsImFwa'
                           'V91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiNz'
                           'E4NDgxMmQtYTMzMC00ZTUzLTlkNDAtYWNkZTUzODExZmM4In0=')
    logger = neptune
    with neptune.create_experiment(name='qmarl', params=params_json):
        neptune.append_tag('agents-{}'.format(params.nb_agents),
                           'height-{}'.format(params.env_height),
                           'width-{}'.format(params.env_width),
                           'erb-{}'.format(1),
                           'target-{}'.format(1),
                           'num_reads-{}'.format(50),
                           'parameter-sharing-{}'.format(0))
        run_experiment(logger, params)


def run_experiment(logger, params):

    agents = []

    env = make_env(params.nb_agents, params.env_height, params.env_width)
    observation_space = env.observation_space()
    action_space = env.action_space()

    for i in range(params.nb_agents):
        agents.append(make_qbm_agent(observation_space, action_space, params.n_layers, params.n_hidden, params.lr))

    training.train(env, agents, params.nb_episodes, params.nb_steps, logger)

if __name__ == '__main__':
    args = sys.argv
    seed = 1589174148213878
    for run in range(5):
        train()
