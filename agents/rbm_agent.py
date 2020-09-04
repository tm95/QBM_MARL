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
        self.epsilon_decay = 0.0001
        self.epsilon_min = 0.1
        self.beta = 0.99

    def tau(self, s, a):
        return np.dot(self.w, s) + np.dot(self.u, a)

    def lam(self, s, a):
        return -logexp_vec(self.tau(s, a))

    def q(self, s, a):
        ph = sig(self.tau(s, a))
        ac = np.sum(self.w * s * ph)

        #TODO: Stimmt hier ac?!
        b = np.sum(self.w * ac * ph)
        c = np.nansum(ph*np.log(ph) + (1-ph)*np.log(1-ph))
        q = ac + b - (1/self.beta)*c
        #print (q)
        #q = -np.sum(self.lam(s, a))
        #print (q)
        return q

    def play(self, s, n_sample, beta):
        # First deterministic initialization
        h = (sig_vec(beta * np.dot(self.w, s)))
        a = (sig_vec(beta * np.dot(self.u.T, h)))

        # Gibbs sampling
        for i in range(n_sample):
            h = (sig_vec(beta * self.tau(s, a)))
            a = (sig_vec(beta * np.dot(self.u.T, h)))

        return np.argmax(a)

    def qlearn(self, s1, a1, s2, a2, r, lr):
        self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)

        b = np.zeros(8)
        b[a1] = 1
        a1 = b

        b = np.zeros(8)
        b[a2] = 1
        a2 = b

        # q learning with gamma = 0
        ph = sig_vec(self.tau(s1, a1))
        self.w = lr * (r + self.q(s2, a2) - self.q(s1, a1)) * np.outer(ph, s1)
        self.u = lr * (r + self.q(s2, a2) - self.q(s1, a1)) * np.outer(ph, a1)

    def policy(self, state, n_sample, beta):
        if torch.rand(1) < self.epsilon:
            return torch.randint(8, (1,)).item()
        with torch.no_grad():
            a = self.play(state, n_sample, beta)
            return a


def make_rbm_agent(ni, nh):

    agent = RBM_agent(1, ni, 8, 0.7)

    return agent
