import math
import random
from neal import SimulatedAnnealingSampler
import torch.nn as nn

class Test_agent(nn.Module):

	def __init__(self, n_layers, dim_state, dim_action, n_hidden):
		super(Test_agent, self).__init__()

		self.Q_hh, self.Q_vh = self.init_weights(n_layers, dim_state, dim_action, n_hidden)
		self.sampler = SimulatedAnnealingSampler()

		self.epsilon = 1
		self.epsilon_min = 0.1
		self.epsilon_decay = 0.0008

		self.lr = 0.001
		self.discount_factor = 0.8
		self.beta = 2
		self.gamma = 0.5

		self.replica_count = 1
		self.average_size = 10
		self.sample_count = self.replica_count * self.average_size

	def init_weights(self, n_layers, dim_state, dim_action, n_hidden):
		Q_hh = dict()

		len_visible = 8
		# len_visible = dim_state + dim_action + 1 ?!

		hidden = []
		for i in range(n_layers):
			hidden.append([(len_visible + i * n_hidden), (len_visible + (i+1) * n_hidden)])

		for i in tuple(range(dim_state)):
			for j in tuple(range(dim_state, len_visible)):
				Q_hh[(i, j)] = 2 * random.random() - 1

		for i in range(n_layers-1):
			for ii in (tuple(range(hidden[i][0], hidden[i][1]))):
				for jj in tuple(range(hidden[i+1][0], hidden[i+1][1])):
					Q_hh[(ii, jj)] = 2 * random.random() - 1

		for i, j in zip(tuple(range(dim_state, len_visible)), tuple(range(hidden[-1][0], hidden[-1][1]))):
			Q_hh[(i, j)] = 2 * random.random() - 1

		Q_vh = dict()
		# Fully connection between state and blue nodes
		for j in (tuple(range(dim_state)) + tuple(range(hidden[-1][0], hidden[-1][1]))):
			for i in range(dim_state):
				Q_vh[(i, j,)] = 2 * random.random() - 1

		# Fully connection between action and red nodes
		for j in (tuple(range(dim_state, len_visible)) + tuple(range(hidden[0][0], hidden[0][1]))):
			for i in range(dim_state, dim_state + dim_action):
				Q_vh[(i, j,)] = 2 * random.random() - 1

		return Q_hh, Q_vh

	def get_3d_hamiltonian_average_value(self, samples, Q):
		i_sample = 0
		h_sum = 0
		w_plus = math.log10(math.cosh(self.gamma * self.beta / self.replica_count) / math.sinh(self.gamma * self.beta / self.replica_count)) / (2 * self.beta)

		for _ in range(self.average_size):
			new_h_0 = new_h_1 = 0
			j_sample = i_sample
			a = i_sample + self.replica_count - 1

			while j_sample < a:
				added_set = set()

				for k_pair, v_weight in Q.items():
					if k_pair[0] == k_pair[1]:
						new_h_0 = new_h_0 + v_weight * (-1 if samples[j_sample][k_pair[0]] == 0 else 1)
					else:
						if k_pair not in added_set and (k_pair[1], k_pair[0]) not in added_set:
							new_h_0 = new_h_0 + v_weight * (-1 if samples[j_sample][k_pair[0]] == 0 else 1) * (-1 if samples[j_sample][k_pair[1]] == 0 else 1)
							added_set.add(k_pair)

				for node_index in samples[j_sample].keys():
					new_h_1 = new_h_1 + (-1 if samples[j_sample][node_index] == 0 else 1) * (-1 if samples[j_sample + 1][node_index] == 0 else 1)
				j_sample += 1

			added_set = set()

			for k_pair, v_weight in Q.items():
				if k_pair[0] == k_pair[1]:
					new_h_0 = new_h_0 + v_weight * (-1 if samples[j_sample][k_pair[0]] == 0 else 1)
				else:
					if k_pair not in added_set and (k_pair[1], k_pair[0], ) not in added_set:
						new_h_0 = new_h_0 + v_weight * (-1 if samples[j_sample][k_pair[0]] == 0 else 1) * (-1 if samples[j_sample][k_pair[1]] == 0 else 1)
						added_set.add(k_pair)

			for node_index in samples[j_sample].keys():
				new_h_1 = new_h_1 + (-1 if samples[j_sample][node_index] == 0 else 1) * (-1 if samples[i_sample][node_index] == 0 else 1)

			h_sum = h_sum + new_h_0 / self.replica_count + w_plus * new_h_1
			i_sample += self.replica_count

		return -1 * h_sum / self.average_size

	def get_free_energy(self, average_hamiltonian, samples):

		key_list = sorted(samples[0].keys())
		prob_dict = dict()

		for i_sample in range(0, len(samples), self.replica_count):
			c_iterable = list()

			for s in samples[i_sample: i_sample + self.replica_count]:
				for k in key_list:
					c_iterable.append(s[k])

			c_iterable = tuple(c_iterable)

			if c_iterable in prob_dict:
				prob_dict[c_iterable] += 1
			else:
				prob_dict[c_iterable] = 1

		a_sum = 0
		div_factor = len(samples) // self.replica_count

		for c in prob_dict.values():
			a_sum = a_sum + c * math.log10(c / div_factor) / div_factor

		return average_hamiltonian + a_sum / self.beta

	def qlearn(self, samples, r, q1, q0, visible_iterable):
		self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)

		prob_dict = self.get_average_configuration(samples)

		for k_pair in self.Q_hh.keys():
			self.Q_hh[k_pair] = self.Q_hh[k_pair] - self.lr * (r + self.discount_factor * q1 - q0) * prob_dict[k_pair] / len(samples)

		for k_pair in self.Q_vh.keys():
			self.Q_vh[k_pair] = self.Q_vh[k_pair] - self.lr * (r + self.discount_factor * q1 - q0) * visible_iterable[k_pair[0]] * prob_dict[k_pair[1]] / len(samples)

	def get_average_configuration(self, samples):
		prob_dict = dict()

		for s in samples:
			for k_pair in self.Q_hh.keys():
				if k_pair in prob_dict:
					prob_dict[k_pair] += (-1 if s[k_pair[0]] == 0 else 1) * (-1 if s[k_pair[1]] == 0 else 1)
				else:
					prob_dict[k_pair] = (-1 if s[k_pair[0]] == 0 else 1) * (-1 if s[k_pair[1]] == 0 else 1)

			for k in s.keys():
				if k in prob_dict:
					prob_dict[k] += (-1 if s[k] == 0 else 1)
				else:
					prob_dict[k] = (-1 if s[k] == 0 else 1)
		return prob_dict

	def create_general_Q_from(self, visible_iterable):
		Q = dict()

		for k_pair, w in self.Q_hh.items():
			Q[k_pair] = Q[(k_pair[1], k_pair[0])] = w

		for k_pair, w in self.Q_vh.items():

			if (k_pair[1], k_pair[1],) not in Q:
				Q[(k_pair[1], k_pair[1])] = w * visible_iterable[k_pair[0]]
			else:
				Q[(k_pair[1], k_pair[1])] += w * visible_iterable[k_pair[0]]

		return Q

	def calculate_q(self, vis_iterable):
		general_Q = self.create_general_Q_from(vis_iterable)
		samples = list(self.sampler.sample_qubo(general_Q, num_reads=self.sample_count).samples())
		random.shuffle(samples)
		hamiltonian = self.get_3d_hamiltonian_average_value(samples, general_Q)
		current_F = self.get_free_energy(hamiltonian, samples)
		return current_F, samples, vis_iterable

	def policy(self, current_state, available_actions, available_actions_list):
		max_tuple = None
		if random.random() > self.epsilon:
			for action_index in available_actions:
				vis_iterable = current_state[1] + available_actions_list[action_index]
				current_F, samples, vis_iterable = self.calculate_q(vis_iterable)
				if max_tuple is None or max_tuple[0] < current_F:
					max_tuple = (current_F, action_index, samples, vis_iterable)
		else:
			action_index = random.choice(tuple(available_actions))
			vis_iterable = current_state[1] + available_actions_list[action_index]
			current_F, samples, vis_iterable = self.calculate_q(vis_iterable)
			max_tuple = (current_F, action_index, samples, vis_iterable)

		return (max_tuple[0], max_tuple[2], max_tuple[3], max_tuple[1], current_state[0])


def make_test_agent(observation_space, action_space):
	agent = Test_agent(4, observation_space, action_space, 4)
	return agent
