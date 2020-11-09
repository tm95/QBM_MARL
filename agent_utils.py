import numpy as np
import copy
from dimod.reference.samplers import ExactSolver
from neal import SimulatedAnnealingSampler
import random
import math
from collections import namedtuple
from time import sleep
import matplotlib.pyplot as plt
import pickle

def get_3d_hamiltonian_average_value(samples, Q, replica_count, average_size, big_gamma, beta):
	'''
	It produces the average Hamiltonian of one dimension higher.

	samples
		It is a list containg the samples from the DWAVE API.

	Q
		It is a dict containg the weights of the Chimera graph.

	replica_count
		It contains the number of replicas in the Hamiltonian of one dimension higher.

	average_size
		It contains the number of configurations of the Hamiltonian of one dimension higher
		used for extracting the value.

	big_gamma, beta
		The parameters with the signification given in the paper.
	'''
	i_sample = 0

	h_sum = 0

	w_plus =\
		math.log10(
			math.cosh( big_gamma * beta / replica_count )\
				/ math.sinh( big_gamma * beta / replica_count )
		) / ( 2 * beta )

	for _ in range(average_size):

		new_h_0 = new_h_1 = 0

		j_sample = i_sample

		a = i_sample + replica_count - 1

		while j_sample < a:

			added_set = set()

			for k_pair, v_weight in Q.items():

				if k_pair[0] == k_pair[1]:

					new_h_0 = new_h_0 + v_weight * ( -1 if samples[j_sample][k_pair[0]] == 0 else 1 )

				else:

					if k_pair not in added_set and ( k_pair[1] , k_pair[0] , ) not in added_set:
					# if True:

						new_h_0 = new_h_0 + v_weight\
							* ( -1 if samples[j_sample][k_pair[0]] == 0 else 1 )\
							* ( -1 if samples[j_sample][k_pair[1]] == 0 else 1 )

						added_set.add( k_pair )

			for node_index in samples[j_sample].keys():

				new_h_1 = new_h_1\
					+ ( -1 if samples[j_sample][node_index] == 0 else 1 )\
					* ( -1 if samples[j_sample + 1][node_index] == 0 else 1 )

			j_sample += 1

		added_set = set()

		for k_pair, v_weight in Q.items():

			if k_pair[0] == k_pair[1]:

				new_h_0 = new_h_0 + v_weight * ( -1 if samples[j_sample][k_pair[0]] == 0 else 1 )

			else:

				if k_pair not in added_set and ( k_pair[1] , k_pair[0] , ) not in added_set:
				# if True:

					new_h_0 = new_h_0 + v_weight\
						* ( -1 if samples[j_sample][k_pair[0]] == 0 else 1 )\
						* ( -1 if samples[j_sample][k_pair[1]] == 0 else 1 )

					added_set.add( k_pair )


		for node_index in samples[j_sample].keys():

			new_h_1 = new_h_1\
				+ ( -1 if samples[j_sample][node_index] == 0 else 1 )\
				* ( -1 if samples[i_sample][node_index] == 0 else 1 )

		h_sum = h_sum + new_h_0 / replica_count + w_plus * new_h_1

		i_sample += replica_count

	return -1 * h_sum / average_size

def get_free_energy(average_hamiltonina, samples, replica_count, beta):
	'''
	It calculates the free energy after the formula in the paper.

	average_hamiltonina
		It is the value of the average of the Hamiltonians of one dimension higher. It is
		created by calling "get_3d_hamiltonian_average_value".

	samples
		It is created by calling the DWAVE API.

	replica_count
		It is the number of replicas in the Hamiltonian of one dimension higher.

	beta
		Parameter presented in the paper.
	'''

	key_list = sorted(samples[0].keys())

	prob_dict = dict()

	for i_sample in range(0,len(samples),replica_count):

		c_iterable = list()

		for s in samples[i_sample : i_sample + replica_count]:
			for k in key_list:
				c_iterable.append( s[k] )

		c_iterable = tuple(c_iterable)

		if c_iterable in prob_dict:
			prob_dict[c_iterable] += 1
		else:
			prob_dict[c_iterable] = 1

	a_sum = 0

	div_factor = len(samples) // replica_count

	for c in prob_dict.values():
		a_sum = a_sum\
			+ c * math.log10( c / div_factor ) / div_factor

	return average_hamiltonina + a_sum / beta

def update_weights(Q_hh, Q_vh, samples, reward, future_F, current_F, visible_iterable,\
	learning_rate, small_gamma):
	'''
	Q_hh
		Contains key pairs (i,j) where i < j

	Q_vh
		Contains key pairs (visible, hidden)

	samples
		It is created by calling the DWAVE API.

	reward
		It is the reward that from either the Environment Network or
		directly from MonALISA.

	future_F
		It is the reward the agent gets at moment t + 1.

	current_F
		It is the reward the agent gets at moment t.

	visible_iterable
		It is the visible units -1/1 iterable the agent uses at moment t.

	learning_rate
		It is the learning rate used in the TD(0) algorithm.

	small_gamma
		It is the discount factor used in the TD(0) algorithm.
	'''
	prob_dict = dict()

	for s in samples:
		for k_pair in Q_hh.keys():
			if k_pair in prob_dict:
				prob_dict[k_pair] +=\
					( -1 if s[k_pair[0]] == 0 else 1 )\
					* ( -1 if s[k_pair[1]] == 0 else 1 )
			else:
				prob_dict[k_pair] =\
					( -1 if s[k_pair[0]] == 0 else 1 )\
					* ( -1 if s[k_pair[1]] == 0 else 1 )

		for k in s.keys():
			if k in prob_dict:
				prob_dict[k] +=\
					( -1 if s[k] == 0 else 1 )
			else:
				prob_dict[k] = ( -1 if s[k] == 0 else 1 )

	for k_pair in Q_hh.keys():
		Q_hh[k_pair] = Q_hh[k_pair] - learning_rate\
			* ( reward + small_gamma * future_F - current_F )\
			* prob_dict[k_pair] / len(samples)

	for k_pair in Q_vh.keys():
		Q_vh[k_pair] = Q_vh[k_pair] - learning_rate\
			* ( reward + small_gamma * future_F - current_F )\
			* visible_iterable[k_pair[0]]\
			* prob_dict[k_pair[1]] / len(samples)

	return Q_hh, Q_vh

def create_general_Q_from(Q_hh, Q_vh, visible_iterable):
	'''
	Creates a weight dict that can be used with the DWAVE API. As the
	visible units are clamped, they are incorporated into biases.

	Q_hh
		Contains key pairs (i,j) where i < j for hidden-hidden weights.

	Q_vh
		Contains key pairs (visible, hidden) for visible-hidden weights.

	visible_iterable
		Contains -1/1 values.
	'''
	Q = dict()

	for k_pair, w in Q_hh.items():
		Q[k_pair] = Q[(k_pair[1], k_pair[0],)] = w

	for k_pair, w in Q_vh.items():

		if (k_pair[1],k_pair[1],) not in Q:
			Q[(k_pair[1],k_pair[1],)] = w * visible_iterable[k_pair[0]]
		else:
			Q[(k_pair[1],k_pair[1],)] += w * visible_iterable[k_pair[0]]

	return Q
