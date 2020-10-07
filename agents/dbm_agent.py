import torch.nn as nn
import torch
import numpy as np
import neal
from random import random
from collections import Counter

def sig(x):
    return 1 / (1 + np.exp(-x))

sig_vec = np.vectorize(sig)

def samp(p):
    if random() < p:
        return 1
    else:
        return 0
samp_vec = np.vectorize(samp)


class DBM_agent(nn.Module):
    def __init__(self,  n_hidden, dim_state, dim_action, n_layers, scale=None):
        super(DBM_agent, self).__init__()

        self.n_layers = n_layers
        self.hidden_layers = int(self.n_layers)
        self.scale = scale

        self.n_hidden = n_hidden
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.state_layer = dim_state
        self.action_layer = dim_action
        self.scale = scale
        self.w = np.random.uniform(low=-self.scale, high=self.scale, size=(n_hidden, dim_state))
        self.u = np.random.uniform(low=-self.scale, high=self.scale, size=(n_hidden, dim_action))
        self.hh = np.random.uniform(low=-self.scale, high=self.scale, size=(n_layers, n_hidden, n_hidden))
        self.num_reads = 100
        self.epsilon = 1
        self.epsilon_decay = 0.001
        self.epsilon_min = 0.1
        self.beta = 0.99
        self.lr = 0.0008
        self.discount_factor = 0.98

        self.sampler = neal.SimulatedAnnealingSampler()

    # Calculate Q-value depending on state, action, hidden nodes and prob of c. Returns negative free energy
    def q(self, s, a):
        hh, p = self.anneal()

        s_energy = []
        a_energy = []
        hidden = []
        h_energy = []

        # Energy action and state
        for k in range(self.n_hidden):
            s_energy.append(np.nansum(np.dot(self.w[k], s) * hh[0][k]))
            a_energy.append(np.nansum(np.dot(self.u[k], a) * hh[-1][k]))

        # Energy Hidden to Hidden
        for l in range(self.n_layers):
            for k in range(self.n_hidden):
                hidden.append(np.nansum((np.dot(self.hh[l][k], hh[l][k]))))
        hh_energy = np.nansum(hidden)

        # Energy Probability
        for i in range(len(p)):
            h_energy.append((p[i]/self.num_reads)*np.log((p[i]/self.num_reads)))
        q = -np.nansum(s_energy) - np.nansum(a_energy) - hh_energy + (1/self.beta) * np.sum(h_energy)

        return -q

    # Convert DBM to QUBO
    def dbm_to_qubo(self):
        Q = {}

        # Hidden to Hidden
        for i in range(self.n_layers):
            for j in range(self.n_hidden):
                for k in range(self.n_hidden):
                    s1 = str(i+1) + str(j)
                    s2 = str(i+2) + str(k)
                    Q[(s1, s2)] = self.hh[i][j][k]

        return Q

    # Convert QUBO structure to DBM
    def qubo_to_dbm(self, Q):
        hh = []

        # Hidden
        for i in range(self.n_layers):
            s = []
            for j in range(self.n_hidden):
                s1 = str(i+1) + str(j)
                s.append(Q[s1])
            hh.append(s)

        return hh

    # Updating weights depending on action and state for time-step 1 and 2.
    def qlearn(self, s1, a1, r):
        self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)

        if a1 == 0:
            a1 = [0, 0]
        elif a1 == 1:
            a1 = [1, 0]
        elif a1 == 2:
            a1 = [0, 1]
        elif a1 == 3:
            a1 = [1, 1]

        # q learning with gamma = 0
        hh, p = self.anneal()
        self.w += self.lr * (r - self.discount_factor * self.q(s1, a1)) * np.outer(hh[0], s1)
        self.u += self.lr * (r - self.discount_factor * self.q(s1, a1)) * np.outer(hh[-1], a1)

        for i in range(self.n_layers-1):
            self.hh[i] += self.lr * (r - self.discount_factor * self.q(s1, a1)) * np.outer(hh[i], hh[i+1])

    # Annealing process. Convert DBM to QUBO, anneal and convert back. Returns averaged Hidden nodes and prob of c
    def anneal(self):
        Q = self.dbm_to_qubo()
        hidden = []
        probs = []

        sampleset = self.sampler.sample_qubo(Q, num_reads=self.num_reads, vartype=0)

        for sample in sampleset:
            hh = self.qubo_to_dbm(sample)
            hidden.append(hh)
            probs.append("".join(str(x) for x in np.array(hh).flatten()))

        p = list(Counter(probs).values())

        # Average over reads
        hidden = np.average(np.array(hidden), axis=0)
        for j in range(self.n_layers):
            for i in range(self.n_hidden):
                if hidden[j][i] > 0.5:
                    hidden[j][i] = 1
                else:
                    hidden[j][i] = 0

        return hidden, p

    # Epsilon-Greedy Policy
    def policy(self, state, beta):
        if torch.rand(1) < self.epsilon:
            return torch.randint(4, (1,)).item()
        with torch.no_grad():
            q = []
            q.append(self.q(state, [0, 0]))
            q.append(self.q(state, [1, 0]))
            q.append(self.q(state, [0, 1]))
            q.append(self.q(state, [1, 1]))

            return np.argmax(q).item()


def make_dbm_agent(ni, nh):
    agent = DBM_agent(4, ni, 2, 10, 0.7)
    return agent
