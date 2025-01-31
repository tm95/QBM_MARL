from __future__ import division
from math import exp, log
from random import random
import numpy as np
import torch

def sig(x):
        return 1 / (1 + np.exp(-x))


sig_vec = np.vectorize(sig)

def samp(p):
    if random() < p:
        return 1
    else:
        return 0
samp_vec = np.vectorize(samp)


def logexp(x):
    if x > 700:
        return x
    else:
        return log(1 + exp(x))


logexp_vec = np.vectorize(logexp)


def safe_log(x):
    if x < 1e-32:
        x = 1e-32
    return log(x)


log_vec = np.vectorize(safe_log)


def similarity(m, y):
    """
    Input
    -------
    m: binary matrix, shape = (data_len, n_feature)
    y: binary matrix, shape = (n_feature)

    Output
    ---------
    similarity between each row and y, the number of same entries over each row and y
    """
    return np.sum((m + y + 1) % 2, axis=1)

class RBM_agent:

    def __init__(self, n_hidden, dim_state, dim_action, scale=None):
        self.n_hidden = n_hidden
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.n_visible = dim_state + dim_action
        self.scale = scale
        self.w = np.random.uniform(low=-self.scale, high=self.scale, size=(n_hidden, dim_state))
        self.u = np.random.uniform(low=-self.scale, high=self.scale, size=(n_hidden, dim_action))
        self.epsilon = 1
        self.epsilon_decay = 0.0008
        self.epsilon_min = 0.01
        self.lr = 0.001
        self.beta = 1

    def activation(self, s, a):
        x = np.dot(self.w, s) + np.dot(self.u, a)
        h = sig(x)
        return h

    def q(self, s, a):

        h = self.activation(s, a)
        e = []
        for k in range(self.n_hidden):
            s_energy = np.nansum(np.dot(self.w[k], s) * h[k])
            a_energy = np.nansum(np.dot(self.u[k], a) * h[k])
            e.append(s_energy + a_energy)
        h_energy = np.nansum([(h[i]*np.log(h[i]) + (1-h[i])*np.log(1-h[i])) for i in range(self.n_hidden)])
        q = np.nansum(e) - (1/self.beta)*h_energy
        return q, h

    def tau(self, s, a):
        return np.dot(self.w, s) + np.dot(self.u, a)

    def lam(self, s, a):
        return -logexp_vec(self.tau(s, a))


    def policy(self, s, beta):
        # First deterministic initialization
        h = samp_vec(sig_vec(beta * np.dot(self.w, s)))
        a = samp_vec(sig_vec(beta * np.dot(self.u.T, h)))

        # Gibbs sampling
        for i in range(10):
            h = samp_vec(sig_vec(beta * self.tau(s, a)))
            a = samp_vec(sig_vec(beta * np.dot(self.u.T, h)))

        q, h = self.q(s, a)

        a = int("".join(str(x) for x in a), 2)

        return a, q, h

    def qlearn(self, s1, a1, r, s2, lr, q, hh):
        self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)

        if a1 == 0:
            a1 = [0, 0]
        elif a1 == 1:
            a1 = [1, 0]
        elif a1 == 2:
            a1 = [0, 1]
        elif a1 == 3:
            a1 = [1, 1]

        self.w += lr * (r - q) * np.outer(hh, s1)
        self.u += lr * (r - q) * np.outer(hh, a1)


def make_rbm_agent(ni, nh):

    agent = RBM_agent(13, ni, nh, 0.7)

    return agent
