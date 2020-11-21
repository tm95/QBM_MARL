import random

class Env():

	def __init__(self):
		super(Env, self).__init__()

		self.reward_function_tuple = self.get_reward_function_tuple()
		self.available_state_dict = self.get_available_state_dict()
		self.available_actions_list = self.get_available_actions_list()
		self.available_actions_per_position_tuple = self.get_available_actions_per_position_tuple()

	def get_optimal_policy_tuple(self):
		optimal_policy_tuple = (
			((4,), (3,), (3,), (3,), (3,)),
			((0,), (3, 0), tuple(), (0,), (3, 0,)),
			((0,), (3, 0), (3,), (0,), (3, 0,)))
		return optimal_policy_tuple

	def get_reward_function_tuple(self):
		reward_function_tuple = (
			(220, 200, 180, 160, 140),
			(200, 180, 160, 120, 120),
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

	def get_available_actions_per_position_tuple(self):
		available_actions_per_position_tuple = (
			((1, 2, 4), (1, 2, 3, 4), (1, 3, 4), (1, 2, 3, 4), (2, 3, 4)),
			((0, 1, 2, 4), (0, 2, 3, 4), tuple(), (0, 1, 2, 4), (0, 2, 3, 4)),
			((0, 1, 4), (0, 1, 3, 4), (1, 3, 4), (0, 1, 3, 4), (0, 3, 4)))
		return available_actions_per_position_tuple

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

		if action == 0:
			new_position = (current_state[0][0] - 1, current_state[0][1])

		elif action == 1:
			new_position = (current_state[0][0], current_state[0][1] + 1)

		elif action == 2:
			new_position = (current_state[0][0] + 1, current_state[0][1])

		elif action == 3:
			new_position = (current_state[0][0], current_state[0][1] - 1)

		else:
			new_position = current_state[0]

		return (new_position, self.available_state_dict[new_position]), self.get_available_actions((new_position, self.available_state_dict[new_position])),\
			self.get_fidelity(action, old_position_tuple), self.get_reward(new_position)

	def get_fidelity(self, action, old_position_tuple):
		fidelity = (1 if action in self.get_optimal_policy_tuple()[old_position_tuple[0]][old_position_tuple[1]] else 0)
		return fidelity

	def get_reward(self, agent_state_tuple):
		if agent_state_tuple == (0, 0):
			reward = self.reward_function_tuple[agent_state_tuple[0]][agent_state_tuple[1]]
		else:
			reward = 0
		#reward = self.reward_function_tuple[agent_state_tuple[0]][agent_state_tuple[1]]
		return reward

	def get_available_actions(self, current_state):
		return filter(lambda e: True, self.available_actions_per_position_tuple[current_state[0][0]][current_state[0][1]])

	def reset(self):
		state = random.choice(tuple(filter(lambda e: e[0] != (0, 0) and e[0] != (1, 2), self.available_state_dict.items())))
		return state, self.get_available_actions(state), self.available_actions_list

	def observation_space(self):
		return len(list(self.get_available_state_dict().values())[0])

	def action_space(self):
		return len(self.get_available_actions_list()[0])

def make_env():
	env = Env()
	return env
