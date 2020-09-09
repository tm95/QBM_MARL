import torch.nn as nn
import torch.nn.functional as F
from agents.rbm_agent import RBM_agent
import torch
import numpy as np


def sig(x):
    return 1 / (1 + np.exp(-x))


sig_vec = np.vectorize(sig)


class QBM_agent(nn.Module):
    def __init__(self,  n_hidden, dim_state, dim_action, n_layers, scale=None):
        super(QBM_agent, self).__init__()

        self.n_layers = len(n_hidden)
        self.rbm_layers = []
        self.hidden_layers = int(self.n_layers)

        self.n_hidden = [n_layers, dim_state]
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.n_visible = dim_state + dim_action
        self.scale = scale
        self.w = np.random.uniform(low=-self.scale, high=self.scale, size=(n_hidden, dim_state))
        self.u = np.random.uniform(low=-self.scale, high=self.scale, size=(n_hidden, dim_action))
        self.hh = np.random.uniform(low=-self.scale, high=self.scale, size=(n_layers, n_hidden, n_hidden))
        self.epsilon = 1
        self.epsilon_decay = 0.0001
        self.epsilon_min = 0.1
        self.beta = 0.99

    def q(self, s, a):
        h, hh, ph = self.anneal(s, a)
        e = []
        hidden = []
        for k in range(self.n_hidden):
            s_energy = np.nansum(np.dot(self.w[k], s) * h[k])
            a_energy = np.nansum(np.dot(self.u[k], a) * h[k])
            e.append(-s_energy - a_energy)
        for l in range(self.n_layers):
            hidden.append([(np.dot(self.hh[l][k], hh[l][k]) for k in range(self.n_hidden))])
        hh_energy = np.nansum(hidden)
        h_energy = np.nansum([(hh[i]*np.log2(hh[i])) for i in range(self.n_hidden)])
        q = np.nansum(e) - hh_energy + (1/self.beta) * h_energy
        return q

    def qlearn(self, s1, a1, s2, a2, r, lr):
        self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)

        b = np.zeros(8)
        b[a1] = 1
        a1 = b

        b = np.zeros(8)
        b[a2] = 1
        a2 = b

        # q learning with gamma = 0
        h, hh, ph = self.anneal()
        self.w += lr * (r + self.q(s2, a2) - self.q(s1, a1)) * np.outer(ph, s1)
        self.u += lr * (r + self.q(s2, a2) - self.q(s1, a1)) * np.outer(ph, a1)
        for i in range(self.n_layers-1):
            self.hh[i] += lr * (r + self.q(s2, a2) - self.q(s1, a1)) * np.outer(hh[i], hh[i+1])

    def anneal(self, s, a):
        # TODO
        h, hh, ph = 0
        return h, hh, ph

    def policy(self, state):
        state = torch.Tensor(state)
        if torch.rand(1) == self.epsilon:
            return torch.randint(self.number_of_actions, (1,)).item()
        with torch.no_grad():
            o = self.calculate_free_energy(state)
            return np.argmax(o).item()


def make_dbm_agent(ni, nh):

    agent = QBM_agent(ni, nh)

    return agent