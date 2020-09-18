import torch.nn as nn
import torch
import numpy as np
import neal
import pyqubo

def sig(x):
    return 1 / (1 + np.exp(-x))


sig_vec = np.vectorize(sig)


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
        self.epsilon = 1
        self.epsilon_decay = 0.0001
        self.epsilon_min = 0.1
        self.beta = 0.99
        self.samples = 150
        self.lr = 0.001

        self.sampler = neal.SimulatedAnnealingSampler()

        self.dbm = []
        self.dbm.append(self.state_layer)
        for i in range(self.n_layers):
            self.dbm.append(self.n_hidden)
        self.dbm.append(self.dim_action)

    def q(self, s, a):
        h = self.activation(s, a)
        e = []
        for k in range(self.n_hidden):
            s_energy = np.nansum(np.dot(self.w[k], s) * h[k])
            a_energy = np.nansum(np.dot(self.u[k], a) * h[k])
            e.append(s_energy + a_energy)
        h_energy = np.nansum([(h[i]*np.log(h[i]) + (1-h[i])*np.log(1-h[i])) for i in range(self.n_hidden)])
        q = np.nansum(e) - (1/self.beta)*h_energy
        return q


    def q(self, s, a):
        h, hh, ph = self.anneal(s)
        e = []
        hidden = []

        #TODO: Überprüfen

        # Energy action and state
        for k in range(self.n_hidden):
            s_energy = np.nansum(np.dot(self.w[k], s) * h[k])
            a_energy = np.nansum(np.dot(self.u[k], a) * h[k])
            e.append(-s_energy - a_energy)

        # Energy Hidden to Hidden
        for l in range(self.n_layers):
            hidden.append([(np.dot(self.hh[l][k], hh[l][k]) for k in range(self.n_hidden))])
        hh_energy = np.nansum(hidden)

        # Energy Probability
        h_energy = np.nansum([(h[i]*np.log2(h[i])) for i in range(self.n_hidden)])
        q = np.nansum(e) - hh_energy + (1/self.beta) * h_energy

        return q

    def dbm_to_qubo(self, s):
        Q = {}

        # Dim State to Hidden
        for i in range(self.dim_state):
            for j in range(self.n_hidden):
                s1 = str(0) + str(i)
                s2 = str(1) + str(j)
                Q[(s1, s2)] = self.w[j][i] * s[i]

        # Hidden to Hidden
        for i in range(self.n_layers):
            for j in range(self.n_hidden):
                for k in range(self.n_hidden):
                    s1 = str(i+1) + str(j)
                    s2 = str(i+2) + str(k)
                    Q[(s1, s2)] = self.hh[i][j][k]

        # Hidden to Action
        for i in range(self.dim_action):
            for j in range(self.n_hidden):
                s1 = str(self.n_layers+1) + str(j)
                s2 = str(self.n_layers+2) + str(i)
                Q[(s1, s2)] = self.u[j][i]

        return Q

    def qubo_to_dbm(self, Q):
        h = []
        hh = []
        a = []

        # State
        for i in range(self.dim_state):
            s1 = str(0) + str(i)
            h.append(Q[s1])

        # Hidden
        for i in range(self.n_layers):
            s = []
            for j in range(self.n_hidden):
                s1 = str(i+1) + str(j)
                s.append(Q[s1])
            hh.append(s)

        # Action
        for i in range(self.dim_action):
            s1 = str(self.n_layers+2) + str(i)
            a.append(Q[s1])

        return h, hh, a

    def qlearn(self, s1, a1, s2, a2, r):
        self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)

        b = np.zeros(self.dim_action)
        b[a1] = 1
        a1 = b

        b = np.zeros(self.dim_action)
        b[a2] = 1
        a2 = b

        # q learning with gamma = 0
        h, hh, ph = self.anneal(s1)
        self.w += self.lr * (r + self.q(s2, a2) - self.q(s1, a1)) * np.outer(h, s1)
        self.u += self.lr * (r + self.q(s2, a2) - self.q(s1, a1)) * np.outer(h, a1)
        for i in range(self.n_layers-1):
            self.hh[i] += self.lr * (r + self.q(s2, a2) - self.q(s1, a1)) * np.outer(hh[i], hh[i+1])

    def anneal(self, s):
        Q = self.dbm_to_qubo(s)
        #TODO: num_reads
        sampleset = self.sampler.sample_qubo(Q, num_reads=125, seed=1234)

        for sample in sampleset:
            h, hh, ph = self.qubo_to_dbm(sample)

        return h, hh, ph

    def policy(self, state, n_sample, beta):
        if torch.rand(1) == self.epsilon:
            return torch.randint(self.number_of_actions, (1,)).item()
        with torch.no_grad():
            h, hh, ph = self.anneal(state)
            print (ph)
            #return np.argmax(ph).item()
            return 2

def make_dbm_agent(ni, nh):

    agent = DBM_agent(25, ni, nh, 4, 0.7)

    return agent