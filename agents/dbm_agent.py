import torch.nn as nn
import torch
import numpy as np
import neal
import random as r
from collections import Counter
import math


class DBM_agent(nn.Module):
    def __init__(self,  n_layers, dim_state, dim_action, n_hidden, scale=None):
        super(DBM_agent, self).__init__()

        self.n_layers = n_layers
        self.hidden_layers = int(self.n_layers)
        self.scale = 0.99

        self.n_hidden = n_hidden
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.state_layer = dim_state
        self.action_layer = dim_action
        self.scale = scale
        self.w = np.random.uniform(low=-self.scale, high=self.scale, size=(n_hidden, dim_state))
        self.u = np.random.uniform(low=-self.scale, high=self.scale, size=(n_hidden, dim_action))
        self.hh = np.random.uniform(low=-self.scale, high=self.scale, size=(n_layers-1, n_hidden, n_hidden))
        self.num_reads = 100
        self.epsilon = 1.3
        self.epsilon_decay = 0.0005
        self.epsilon_min = 0.1
        self.beta = 1.0
        self.lr = 0.001
        self.discount_factor = 1.0
        self.replica_count = 5
        self.average_size = 20

        self.sampler = neal.SimulatedAnnealingSampler()

    # Calculate Q-value depending on state, action, hidden nodes and prob of c. Returns negative free energy
    def qlearn(self, s, a, r, lr, q, hh):
        self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)

        if a == 0:
            a = [0, 0]
        elif a == 1:
            a = [1, 0]
        elif a == 2:
            a = [0, 1]
        elif a == 3:
            a = [1, 1]


        self.w -= self.lr * (r - self.discount_factor * q) * np.outer(hh[0], s)
        self.u -= self.lr * (r - self.discount_factor * q) * np.outer(hh[-1], a)

        for i in range(self.n_layers-1):
            self.hh[i] -= self.lr * (r - self.discount_factor * q) * np.outer(hh[i], hh[i+1])

        return q

    def q(self, s, a):
        hh, p, h_val, samples = self.anneal(s, a)

        q = self.get_free_energy(h_val, samples, 2, 2)

        return q, hh

    # Convert DBM to QUBO
    def dbm_to_qubo(self, state, action):
        Q = {}

        # Hidden to Hidden
        for i in range(self.n_layers-1):
            for j in range(self.n_hidden):
                for k in range(self.n_hidden):
                    s1 = str(i+1) + str(j)
                    s2 = str(i+2) + str(k)
                    Q[(s1, s2)] = Q[(s2, s1)] = self.hh[i][j][k]

        # Hidden Neurons
        for i in range(self.n_layers):
            for j in range(self.n_hidden):
                s1 = str(i + 1) + str(j)
                if i == 0:
                    Q[(s1, s1)] = sum([self.w[j][k] if state[k] == 1 else -self.w[j][k] for k in range(self.dim_state)])
                elif i == (self.n_layers - 1):
                    Q[(s1, s1)] = sum([self.u[j][k] if action[k] == 1 else -self.u[j][k] for k in range(self.dim_action)])

        return Q

    # Convert QUBO structure to DBM
    def qubo_to_dbm(self, Q):
        hh = []

        # Hidden
        for i in range(self.n_layers):
            s = []
            for j in range(self.n_hidden):
                s1 = str(i + 1) + str(j)
                if Q[s1] == 0:
                    s.append(-1)
                elif Q[s1] == 1:
                    s.append(1)
            hh.append(s)

        return hh


    # Annealing process. Convert DBM to QUBO, anneal and convert back. Returns averaged Hidden nodes and prob of c
    def anneal(self, state, action):

        if action == 0:
            action = [0, 0]
        elif action == 1:
            action = [1, 0]
        elif action == 2:
            action = [0, 1]
        elif action == 3:
            action = [1, 1]


        Q = self.dbm_to_qubo(state, action)
        hidden = []
        probs = []

        sample_count = self.replica_count * self.average_size

        sampleset = list(self.sampler.sample_qubo(Q, num_reads=sample_count, vartype=0).samples())
        r.shuffle(sampleset)

        h_val = self.get_3d_hamiltonian_average_value(sampleset, Q, self.replica_count, self.average_size, 0.5, 2)

        for sample in sampleset:
            hh = self.qubo_to_dbm(sample)
            hidden.append(hh)
            probs.append("".join(str(x) for x in np.array(hh).flatten()))

        p = list(Counter(probs).values())

        # Average over reads
        hidden = np.average(np.array(hidden), axis=0)

        return hidden, p, h_val, sampleset

    def get_3d_hamiltonian_average_value(self, samples, Q, replica_count, average_size, big_gamma, beta):

        i_sample = 0
        h_sum = 0

        w_plus = math.log10(math.cosh(big_gamma * beta / replica_count)/ math.sinh(big_gamma * beta / replica_count)) / (2 * beta)

        for _ in range(average_size):
            new_h_0 = new_h_1 = 0
            j_sample = i_sample
            a = i_sample + replica_count - 1

            while j_sample < a:
                added_set = set()

                for k_pair, v_weight in Q.items():
                    if k_pair[0] == k_pair[1]:
                        new_h_0 = new_h_0 + v_weight * (-1 if samples[j_sample][k_pair[0]] == 0 else 1)
                    else:
                        if k_pair not in added_set and (k_pair[1], k_pair[0],) not in added_set:
                            new_h_0 = new_h_0 + v_weight * (-1 if samples[j_sample][k_pair[0]] == 0 else 1)\
                                      * (-1 if samples[j_sample][k_pair[1]] == 0 else 1)
                            added_set.add(k_pair)

                for node_index in samples[j_sample].keys():
                    new_h_1 = new_h_1 + (-1 if samples[j_sample][node_index] == 0 else 1) \
                              * (-1 if samples[j_sample + 1][node_index] == 0 else 1)
                j_sample += 1

            added_set = set()

            for k_pair, v_weight in Q.items():
                if k_pair[0] == k_pair[1]:
                    new_h_0 = new_h_0 + v_weight * (-1 if samples[j_sample][k_pair[0]] == 0 else 1)
                else:
                    if k_pair not in added_set and (k_pair[1], k_pair[0],) not in added_set:
                        new_h_0 = new_h_0 + v_weight * (-1 if samples[j_sample][k_pair[0]] == 0 else 1) \
                                  * (-1 if samples[j_sample][k_pair[1]] == 0 else 1)
                        added_set.add(k_pair)

            for node_index in samples[j_sample].keys():
                new_h_1 = new_h_1 + (-1 if samples[j_sample][node_index] == 0 else 1) \
                          * (-1 if samples[i_sample][node_index] == 0 else 1)

            h_sum = h_sum + new_h_0 / replica_count + w_plus * new_h_1
            i_sample += replica_count

        return -1 * h_sum / average_size


    def get_free_energy(self, average_hamiltonian, samples, replica_count, beta):

        key_list = sorted(samples[0].keys())
        prob_dict = dict()

        for i_sample in range(0, len(samples), replica_count):
            c_iterable = list()

            for s in samples[i_sample: i_sample + replica_count]:
                for k in key_list:
                    c_iterable.append(s[k])

            c_iterable = tuple(c_iterable)

            if c_iterable in prob_dict:
                prob_dict[c_iterable] += 1
            else:
                prob_dict[c_iterable] = 1

        a_sum = 0

        div_factor = len(samples) // replica_count

        for c in prob_dict.values():
            a_sum = a_sum + c * math.log10(c / div_factor) / div_factor

        return average_hamiltonian + a_sum / beta

    # Epsilon-Greedy Policy
    def policy(self, state, beta):
        if torch.rand(1) < self.epsilon:
            a = torch.randint(4, (1,)).item()
            q_val, hh = self.q(state, a)
            return a, q_val, hh

        with torch.no_grad():
            q = []
            hidden = []

            a, hh = self.q(state, [0, 0])
            q.append(a)
            hidden.append(hh)

            a, hh = self.q(state, [1, 0])
            q.append(a)
            hidden.append(hh)

            a, hh = self.q(state, [0, 1])
            q.append(a)
            hidden.append(hh)

            a, hh = self.q(state, [1, 1])
            q.append(a)
            hidden.append(hh)

            #print (q)

            a = np.argmin(q).item()
            hh = hidden[a]
            q_val = q[a]

            return a, q_val, hh

def make_dbm_agent(ni, nh):
    agent = DBM_agent(4, ni, 2, 10, 0.7)
    return agent
