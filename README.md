## About
This is the code for the experiments of our paper [Towards Multi-agent Reinforcement Learning using Quantum Boltzmann Machines](https://arxiv.org/pdf/2109.10900).

## Setup
1. Set up a virtualenv with 
2. Run ```pip install -r requirements.txt``` to get requirements
3. We used neptune.ai for experiment tracking. In [neptune.ai](neptune.ai), create a Neptune project following the [documentation](https://docs.neptune.ai/setup/creating_project/). Add you project name and api_token in [main.py](main.py). Note, that the code includes the experience replay buffer and separation of policy and target network. 

## Remarks
1. We use the config file [params](params.json) to configure the environenments and the RL algorithm for our experiments.
2. Results of the experiments are stored in the neptune.ai project
3. The code for the environment can be found in [env](env) folder
4. The code for the QBM and RBM agent can be found in the [agents](agents) folder

# Citation
If you use this code in your own work, please cite our paper:
```
@article{mueller2022qmarl,
  title={Towards Multi-agent Reinforcement Learning using Quantum Boltzmann Machines},
  author={M\"{u}ller, Tobias and Roch, Christoph and Schmid, Kyrill and Altmann, Philipp},
  booktitle={Proceedings of the 14th International Conference on Agents and Artificial Intelligence - Volume 1: ICAART},
  year={2022}
}
```

# Acknowledgments
Some of this code was based on an implementation of ["Free energy-based reinforcement learning using quantum processor"](https://github.com/Mircea-Marian/attract_grid_data_flow_optimization/tree/master)
