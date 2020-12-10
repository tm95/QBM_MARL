import math
import random
from neal import SimulatedAnnealingSampler
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity: int, seed: int = 42) -> None:
        self.rng = random
        self.rng.seed(seed)
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args) -> None:
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size) -> []:
        return self.rng.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DBM:
	def __init__(self, n_layers, dim_state, dim_action, n_hidden):
		super(DBM, self).__init__()

		self.Q_hh, self.Q_vh = self.init_weights(n_layers, dim_state, dim_action, n_hidden)
		self.sampler = SimulatedAnnealingSampler()

		self.beta = 2
		self.gamma = 0.5

		self.replica_count = 1
		self.average_size = 10
		self.sample_count = self.replica_count * self.average_size

	def init_weights(self, n_layers, dim_state, dim_action, n_hidden):
		Q_hh = dict()

		len_visible = dim_state + dim_action

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


class QBM_agent:
	def __init__(self, dim_state, dim_action, n_layers, n_hidden, lr):
		super(QBM_agent, self).__init__()

		self.policy_net = DBM(n_layers, dim_state, dim_action, n_hidden)
		self.target_net = DBM(n_layers, dim_state, dim_action, n_hidden)

		self.action_size = 5

		self.epsilon = 1
		self.epsilon_min = 0.1
		self.epsilon_decay = 0.0008

		self.lr = 0.005
		self.discount_factor = 0.8

		self.mini_batch_size = 8
		self.warm_up_duration = 250
		self.target_update_period = 250
		self.memory = ReplayMemory(20000, 42)
		self.training_count = 1

	def qlearn(self, state, available_actions_list, next_state, reward):
		if len(self.memory) < self.warm_up_duration:
			return

		transitions = self.memory.sample(self.mini_batch_size)
		batch = Transition(*zip(*transitions))

		batch = [state, available_actions_list, next_state, reward]

		for i in range(1):
			vis_iterable = batch[0] + batch[1]
			current_F, samples, visible_iterable = self.policy_net.calculate_q(vis_iterable)
			prob_dict = self.target_net.get_average_configuration(samples)

			future_F = -100000

			for action_index in range(self.action_size):
				vis_iterable = batch[2][1] + available_actions_list
				F, samples, vis_iterable = self.policy_net.calculate_q(vis_iterable)
				if F > future_F:
					future_F = F

			for k_pair in self.policy_net.Q_hh.keys():
				self.policy_net.Q_hh[k_pair] = self.policy_net.Q_hh[k_pair] - self.lr * (batch[3] + self.discount_factor * future_F - current_F) * prob_dict[k_pair] / len(samples)

			for k_pair in self.policy_net.Q_vh.keys():
				self.policy_net.Q_vh[k_pair] = self.policy_net.Q_vh[k_pair] - self.lr * (batch[3] + self.discount_factor * future_F - current_F) * visible_iterable[k_pair[0]] * prob_dict[k_pair[1]] / len(samples)

		self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)

		self.training_count += 1
		if self.training_count % self.target_update_period == 0:
			self.target_net.Q_hh = self.policy_net.Q_hh
			self.target_net.Q_vh = self.policy_net.Q_vh

	def save(self, state, action, next_state, reward):
		self.memory.push(state, action, next_state, reward)

	def policy(self, current_state, available_actions_list):
		max_tuple = None
		if random.random() > self.epsilon:
			for action_index in range(self.action_size):
				vis_iterable = current_state[1] + available_actions_list[action_index]
				current_F, samples, vis_iterable = self.policy_net.calculate_q(vis_iterable)
				if max_tuple is None or max_tuple[0] < current_F:
					max_tuple = (current_F, action_index, samples, vis_iterable)
		else:
			action_index = random.choice(range(self.action_size))
			max_tuple = (0, action_index, 0, 0)

		return (max_tuple[1])


def make_qbm_agent(observation_space, action_space, n_layers, n_hidden, lr):
	agent = QBM_agent(observation_space, action_space, n_layers, n_hidden, lr)
	return agent
