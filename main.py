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

    logger = neptune.init_run( monitoring_namespace = "monitoring")
    logger["parameters"] = params_json

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
    for run in range(10):
        train()
