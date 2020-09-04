import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from __future__ import division
import time
from math import exp, log, floor
from operator import add, mul
from random import random, randint
import numpy as np
from matplotlib import pylab as plt


def sig(x):
    if x < -700:
        return 0.0
    else:
        return 1 / ( 1 + np.exp(-x) )
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
        return log(1+exp(x))
logexp_vec = np.vectorize(logexp)

def safe_log(x):
    if x < 1e-32:
        x = 1e-32
    return log(x)
log_vec = np.vectorize(safe_log)


class RBM_agent(nn.Module):
    def __init__(self,
                 n_visible=256,
                 n_hidden=64):
        super(RBM_agent, self).__init__()
        self.W = nn.Parameter(
            torch.Tensor(n_hidden, n_visible).uniform_(-1.0 / (n_visible + n_hidden), 1.0 / (n_visible + n_hidden)))
        self.v_bias = nn.Parameter(torch.zeros(n_visible))
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))
        self.epsilon = 1
        self.epsilon_decay = 0.00008
        self.epsilon_min = 0.1
        self.number_of_actions = 8
        self.device = torch.device("cpu")
        self.learning_rate = 0.001
        self.gamma = 0.99

    def sample_from_p(self, p):
        return torch.bernoulli(p)

    def calculate_h(self, v):
        p_h = torch.sigmoid(F.linear(v, self.W, self.h_bias))
        return p_h

    def calculate_free_energy(self, s, a):
        tau = np.dot(self.w, s) + np.dot(self.u, a)
        lam = -logexp_vec(tau)
        q = -np.sum(lam)
        return q


    def train(self, s, a, r, lr):
        # Change to modified quantum training algorithm
        self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)

        ph = sig_vec(self.tau(s, a))
        self.w += lr * (r - self.q(s, a)) * np.outer(ph, s)
        self.u += lr * (r - self.q(s, a)) * np.outer(ph, a)

    def policy(self, state):
        state = torch.Tensor(state)
        if torch.rand(1) == self.epsilon:
            return torch.randint(self.number_of_actions, (1,)).item()
        with torch.no_grad():
            h = self.calculate_h(state)
            #print (h)
            o = self.calculate_free_energy(state, h)
            return np.argmax(h).item()


def make_rbm_agent(ni, nh):

    agent = RBM_agent(ni, nh)

    return agent
