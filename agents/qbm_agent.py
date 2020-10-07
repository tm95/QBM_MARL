import torch.nn as nn
import torch
import numpy as np
import neal

def sig(x):
    return 1 / (1 + np.exp(-x))


sig_vec = np.vectorize(sig)


class QBM_agent(nn.Module):
    def __init__(self,  n_hidden, dim_state, dim_action, n_layers, scale=None):
        super(QBM_agent, self).__init__()

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
        self.num_reads = 10
        self.epsilon = 1
        self.epsilon_decay = 0.0005
        self.epsilon_min = 0.1
        self.beta = 0.99
        self.samples = 10
        self.lr = 0.01
        self.discount_factor = 0.8

        self.sampler = neal.SimulatedAnnealingSampler()

    # Calculate Q-value depending on state, action, hidden nodes and prob of c. Returns negative free energy
    def q(self, s, a):
        hh, energy, p = self.anneal()
        h_energy = []

        # Energy Probability
        for i in range(len(p)):
            h_energy.append((p[i]/self.num_reads)*np.log((p[i]/self.num_reads)))

        q = - energy - (1/self.beta) * np.sum(h_energy)

        return q

    # Convert DBM structure to QUBO
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
    def qlearn(self, s1, a1, s2, a2, r):
        self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)

        b = np.zeros(self.dim_action)
        b[a1] = 1
        a1 = b

        b = np.zeros(self.dim_action)
        b[a2] = 1
        a2 = b

        # q learning with gamma = 0
        hh, e, p = self.anneal()
        self.w += self.lr * (r + self.discount_factor * self.q(s2, a2) - self.q(s1, a1)) * np.outer(hh[0], s1)
        self.u += self.lr * (r + self.discount_factor * self.q(s2, a2) - self.q(s1, a1)) * np.outer(hh[-1], a1)
        for i in range(self.n_layers-1):
            self.hh[i] += self.lr * (r + self.discount_factor * self.q(s2, a2) - self.q(s1, a1)) * np.outer(hh[i], hh[i+1])

    # Annealing process. Convert DBM to QUBO, anneal and convert back. Returns averaged Hidden nodes, H[eff] and prob of c
    def anneal(self):
        Q = self.dbm_to_qubo()
        hidden = []

        sampleset = self.sampler.sample_qubo(Q, num_reads=self.num_reads, seed=1234, vartype=1)

        energy = np.average(sampleset.record.energy)
        p = sampleset.record.num_occurrences

        for sample in sampleset:
            hh = self.qubo_to_dbm(sample)
            hidden.append(hh)

        # Average over reads
        hidden = np.average(np.array(hidden), axis=0)
        for j in range(self.n_layers):
            for i in range(self.n_hidden):
                if hidden[j][i] > 0.5:
                    hidden[j][i] = 1
                else:
                    hidden[j][i] = 0

        return hidden, energy, p

    # Epsilon-Greedy Policy
    def policy(self, state, beta):
        if torch.rand(1) < self.epsilon:
            return torch.randint(self.dim_action, (1,)).item()
        with torch.no_grad():
            q = []
            q.append(self.q(state, [0, 0]))
            q.append(self.q(state, [1, 0]))
            q.append(self.q(state, [0, 1]))
            q.append(self.q(state, [1, 1]))

            return np.argmax(q).item()

def make_qbm_agent(ni, nh):

    agent = QBM_agent(10, ni, nh, 4, 0.7)
    return agent