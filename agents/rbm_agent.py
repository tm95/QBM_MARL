import torch
import torch.nn as nn
from collections import namedtuple
import numpy as np


class SimpleDQN(torch.nn.Module):
    def __init__(self, height, outputs):
        super(SimpleDQN, self).__init__()
        self.pre_head_dim = 32  # 32
        self.fc_net = nn.Sequential(
            nn.Linear(height, 32),  # 64
            nn.ELU(),
            nn.Linear(32, self.pre_head_dim),  # 64
            nn.ELU()
        )

        self.action_head = nn.Linear(self.pre_head_dim, outputs)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc_net(x)
        return self.action_head(x)


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class RBM_Agent(torch.nn.Module):
    def __init__(self, ni, nh):
        super(RBM_Agent, self).__init__()
        self.W = torch.randn(nh, ni) # Init nodes
        self.a = torch.randn(1, nh) # Init bias hidden nodes, 1 is batch size
        self.b = torch.randn(1, ni) # Init bias visible input nodes, 1 is batch size
        self.epsilon_decay = 0.0008
        self.epsilon_min = 0.1
        self.number_of_actions = 5
        self.device = torch.device("cpu")
        self.learning_rate = 0.001
        self.gamma = 0.99

    def sample_h(self, x): # x represents visible neuron
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation) # activation function
        return p_h_given_v, torch.bernoulli(p_h_given_v)

    def sample_vi(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)

    def sample_vo(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)

    def train(self, r, q0, q1, state, a0, a1):
        # Change to modified quantum training algorithm
        self.b += self.learning_rate*(r + self.gamma*q1[:, a1] - q0[:, a0])*self.b[:, state]
        self.a += self.learning_rate*(r + self.gamma*q1[:, a1] - q0[:, a0])*self.a[:, a0]

    def policy(self, state):
        #TODO: policy
        with torch.no_grad():
            state = torch.tensor([state], device=self.device, dtype=torch.float32)
            ph0, _ = self.sample_h(state)
            for i in range(10):
                _, hk = self.sample_h(state)
                _, vk = self.sample_vo(hk)
                vk[state<0] = state[state<0]
            o, _ = self.sample_h(vk)
            return -o
 # TODO: Free energy calculation

def make_rbm_agent(ni, nh):

    agent = RBM_Agent(ni, nh)

    return agent
