import numpy as np

class Env():

	def __init__(self, nb_agents):
		super(Env, self).__init__()

		self.nb_agents = nb_agents
		self.height = 3
		self.width = 3
		self.goals = []

		self.available_states = self.get_available_states()
		self.available_actions_list = self.get_available_actions_list()

	def get_available_states(self):
		available_states = []
		for i in range(self.height):
			for j in range(self.width):
				available_states.append((i, j))
		return available_states

	def get_available_actions_list(self):
		available_actions_list = list()
		i = 0
		for q1 in ((-1, ), (1, ),):
			for q2 in ((-1, ), (1, ),):
				for q3 in ((-1, ), (1, ),):
					if i < 5:
						available_actions_list.append(q1 + q2 + q3)
					else:
						break
					i+=1
				if i >= 5:
					break
			if i >= 5:
				break
		return available_actions_list

	def render(self, x_y_position):
		for i in range(3):
			line_string = ''
			for j in range(5):
				if x_y_position == (i, j,):
					line_string += 'X'
				elif (i, j,) in list(self.goals):
					line_string += 'G'
				else:
					line_string += 'O'
			print(line_string)
		print()

	def step(self, action, current_state):

		next_state = self.get_next_state(action, current_state)
		reward = self.get_reward(next_state)
		done = self.get_done()

		return next_state, reward, done

	def get_next_state(self, action, current_state):
		obs = []

		for i in range(self.nb_agents):
			binary_agent = -np.ones((self.height, self.width), dtype=int)
			binary_others = -np.ones((self.height, self.width), dtype=int)

			if action[i] == 0:
				next_state = (current_state[i][0][0] - 1, current_state[i][0][1])
			elif action[i] == 1:
				next_state = (current_state[i][0][0], current_state[i][0][1] + 1)
			elif action[i] == 2:
				next_state = (current_state[i][0][0] + 1, current_state[i][0][1])
			elif action[i] == 3:
				next_state = (current_state[i][0][0], current_state[i][0][1] - 1)
			else:
				next_state = (current_state[i][0])

			if next_state not in list(self.available_states):
				next_state = (current_state[i][0])

			binary_agent[next_state[0]][next_state[1]] = 1
			if (0, 0) in list(self.goals):
				binary_agent[0][0] = 1

			for j in range(self.nb_agents):
				if i != j:

					if action[j] == 0:
						n = (current_state[j][0][0] - 1, current_state[j][0][1])
					elif action[j] == 1:
						n = (current_state[j][0][0], current_state[j][0][1] + 1)
					elif action[j] == 2:
						n = (current_state[j][0][0] + 1, current_state[j][0][1])
					elif action[j] == 3:
						n = (current_state[j][0][0], current_state[j][0][1] - 1)
					else:
						n = (current_state[j][0])

					if n not in list(self.available_states):
						n = (current_state[j][0])

					binary_others[n[0]][n[1]] = 1
					if (2,2) in list(self.goals):
						binary_others[2][2] = 1

					#binary_others[j][j] = 1
					#binary_agent[n[0]][n[1]] = 1

			#obs.append((next_state, tuple(binary_agent.flatten()) + tuple(binary_others.flatten())))
			obs.append((next_state, tuple(binary_agent.flatten())))

		return obs

	def get_done(self):
		done = []
		for i in range(self.nb_agents):
			if len(self.goals) == 0:
				done.append(True)
			else:
				done.append(False)
		return done

	def get_reward(self, agent_state_tuple):
		rewards = []
		for i in range(self.nb_agents):
			if i == 0:
				if agent_state_tuple[0][0] == (0, 0) and agent_state_tuple[0][0] in list(self.goals):
					rewards.append(220)
					self.goals.remove(agent_state_tuple[i][0])
				elif agent_state_tuple[0][0] in list(self.goals):
					rewards.append(-210)
				elif (0,0) not in list(self.goals):
					rewards.append(0)
				else:
					rewards.append(-10)
			if i == 1:
				if agent_state_tuple[1][0] == (2, 2) and agent_state_tuple[1][0] in list(self.goals):
					rewards.append(220)
					self.goals.remove(agent_state_tuple[i][0])
				elif agent_state_tuple[1][0] in list(self.goals):
					rewards.append(-210)
				elif (2,2) not in list(self.goals):
					rewards.append(0)
				else:
					rewards.append(-10)
#			if agent_state_tuple[i][0] in list(self.goals):
#				rewards.append(220)
#				self.goals.remove(agent_state_tuple[i][0])
#			else:
#				rewards.append(-10)
		return rewards

	def reset(self):
		obs = []
		decimal = [(2,0), (0,2)]
		self.goals = [(0,0), (2,2)]
		for i in range(self.nb_agents):
			#decimal = (np.random.randint(self.height), np.random.randint(self.width))
			binary = -np.ones((self.height, self.width), dtype=int)
			binary[decimal[i][0]][decimal[i][1]] = 1
			binary[self.goals[i][0]][self.goals[i][1]] = 1
			binary_others = -np.ones((self.height, self.width), dtype=int)
			for j in range(self.nb_agents):
				if i != j:
					binary_others[decimal[j][0]][decimal[j][1]] = 1
					binary_others[self.goals[j][0]][self.goals[j][1]] = 1
			#obs.append((decimal[i], tuple(binary.flatten()) + tuple(binary_others.flatten())))
			obs.append((decimal, tuple(binary.flatten())))

		return obs, self.available_actions_list

	def observation_space(self):
		return self.height * self.width * 2

	def action_space(self):
		return len(self.get_available_actions_list()[0])

def make_env(nb_agents):
	env = Env(nb_agents)
	return env
