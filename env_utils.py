import random

class Env():

	def __init__(self):
		super(Env, self).__init__()

		self.reward_function_tuple = self.get_reward_function_tuple()
		self.available_state_dict = self.get_available_state_dict()
		self.available_actions_list = self.get_available_actions_list()

	def get_optimal_policy_tuple(self):
		optimal_policy_tuple = (
			((4,), (3,), (3,), (3,), (3,)),
			((0,), (3, 0), tuple(), (0,), (3, 0,)),
			((0,), (3, 0), (3,), (0,), (3, 0,)))
		return optimal_policy_tuple

	def get_reward_function_tuple(self):
		reward_function_tuple = (
			(220, 200, 180, 160, 140),
			(200, 180, 160, -200, 120),
			(180, 160,   0, 120, 100))
		return reward_function_tuple

	def get_available_state_dict(self):
		available_state_dict = dict()
		i = 0
		for q1 in ((-1,), (1,),):
			for q2 in ((-1,), (1,),):
				for q3 in ((-1,), (1,),):
					for q4 in ((-1,), (1,),):
						if i != 7:
							available_state_dict[(i // 5, i % 5,)] = q1 + q2 + q3 + q4
						i += 1
						if i >= 15: break
					if i >= 15: break
				if i >= 15: break
			if i >= 15: break
		return available_state_dict

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
				elif (i, j,) == (0, 0):
					line_string += 'G'
				elif (i, j) == (1, 2):
					line_string += 'W'
				else:
					line_string += 'O'
			print(line_string)
		print()

	def step(self, action, current_state, old_position_tuple):

		next_state = self.get_next_state(action, current_state)
		done = self.get_done(next_state)
		fidelity = self.get_fidelity(action, old_position_tuple)
		reward = self.get_reward(next_state[0])

		return next_state, fidelity, reward, done

	def get_next_state(self, action, current_state):
		if action == 0:
			next_state = (current_state[0][0] - 1, current_state[0][1])
		elif action == 1:
			next_state = (current_state[0][0], current_state[0][1] + 1)
		elif action == 2:
			next_state = (current_state[0][0] + 1, current_state[0][1])
		elif action == 3:
			next_state = (current_state[0][0], current_state[0][1] - 1)
		else:
			next_state = current_state[0]

		if next_state not in self.get_available_state_dict().keys():
			next_state = current_state[0]

		return (next_state, self.available_state_dict[next_state])

	def get_done(self, next_state):
		if next_state[0] == (0, 0):
			done = True
		else:
			done = False
		return done

	def get_fidelity(self, action, old_position_tuple):
		fidelity = (1 if action in self.get_optimal_policy_tuple()[old_position_tuple[0]][old_position_tuple[1]] else 0)
		return fidelity

	def get_reward(self, agent_state_tuple):
		if agent_state_tuple == (0, 0):
			reward = self.reward_function_tuple[agent_state_tuple[0]][agent_state_tuple[1]]
		#elif agent_state_tuple == (1, 3):
	#		reward = self.reward_function_tuple[agent_state_tuple[0]][agent_state_tuple[1]]
		else:
			reward = -10
		return reward

	def reset(self):
		state = random.choice(tuple(filter(lambda e: e[0] != (0, 0) and e[0] != (1, 2), self.available_state_dict.items())))
		return state, self.available_actions_list

	def observation_space(self):
		return len(list(self.get_available_state_dict().values())[0])

	def action_space(self):
		return len(self.get_available_actions_list()[0])

def make_env():
	env = Env()
	return env
