import torch
import torch.nn as nn
from collections import namedtuple


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
        self.epsilon = 1
        self.number_of_actions = 5
        self.device = torch.device("cpu")

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

    def train(self, v0, vk, ph0, phk): # Change to modified quantum training algorithm
        self.W += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)

    def policy(self, state):
        if torch.rand(1) < self.epsilon:
            self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)
            return torch.randint(self.number_of_actions, (1,)).item()
        with torch.no_grad():
            state = torch.tensor([state], device=self.device, dtype=torch.float32)
            ph0, _ = self.sample_h(state)
            for i in range(10):
                _, hk = self.sample_h(state)
                _, vk = self.sample_vo(hk)
                vk[state<0] = state[state<0]
            o, _ = self.sample_h(vk)
            return o.max(1)[1]


def make_rbm_agent(ni, nh):

    agent = RBM_Agent(ni, nh)

    return agent
