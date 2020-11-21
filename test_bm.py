from agents.rbm_agent import make_rbm_agent
from agents.dbm_agent import make_dbm_agent
import random
import matplotlib.pyplot as plt

def test_update_function(agent):

    '''
    Tests the updating mechanism. The free energies should go
    towards the values in "reward_tuple".
    '''

    reward_tuple = (500, 400, 300, 200, 100)
    tries_count = 400
    available_state_dict = create_state_dict()
    available_actions_list = create_action_dict()
    agent_state_tuple = random.choice(tuple(available_state_dict.items()))

    f_value_list = list()

    for _ in range(tries_count):
        f_value_list.append(list())

        for action_index in range(5):

            current_F, hh = agent.q(agent_state_tuple[1], available_actions_list[action_index])
            agent.qlearn(agent_state_tuple[1], available_actions_list[action_index], reward_tuple[action_index], agent_state_tuple[1], 0.001, current_F, hh)
            f_value_list[-1].append(current_F)

    print_results(f_value_list)

def test_rl(agent):

    # Initialize constants
    game_count = 70

    step_count_list = list()
    fidelity_list = list()

    available_state_dict = create_state_dict()
    available_actions_list = create_action_dict()
    optimal_policy_tuple = create_optimal_policy_tuple()
    reward_function_tuple = create_reward_function_tuple()
    available_actions_per_position = create_available_actions()

    for i_round in range(game_count):
        agent_state_tuple = random.choice(tuple(filter(lambda e: e[0] != (0, 0) and e[0] != (1, 2), available_state_dict.items())))

        print(i_round, ':', 'step =', 0, '; position =', agent_state_tuple[0])

        step_count = 1

        action, current_F, hh = agent.policy_test(agent_state_tuple, 1, available_actions_per_position, available_actions_list)
        old_position_tuple = agent_state_tuple
        next_state = get_next_state(action, agent_state_tuple)

        if next_state in available_state_dict.keys():
            agent_state_tuple = (next_state, available_state_dict[next_state])
        else:
            print ("Next State not in Dict")

        fidelity_count = 1 if action in optimal_policy_tuple[old_position_tuple[0][0]][old_position_tuple[0][1]] else 0

        if agent_state_tuple[0] != (0, 0):
            while True:

                action, future_F, hh = agent.policy_test(agent_state_tuple, 1, available_actions_per_position, available_actions_list)
                old_position_tuple = agent_state_tuple
                next_state = get_next_state(action, agent_state_tuple)
                if next_state in available_state_dict.keys():
                    agent_state_tuple = (next_state, available_state_dict[next_state])
                else:
                    print("Next State not in Dict")

                reward = reward_function_tuple[agent_state_tuple[0][0]][agent_state_tuple[0][1]]

                fidelity_count += 1 if action in optimal_policy_tuple[old_position_tuple[0][0]][old_position_tuple[0][1]] else 0
                step_count += 1

                agent.qlearn(old_position_tuple[1], available_actions_list[action], reward, agent_state_tuple[1], 0.001, future_F, current_F, hh)

                if agent_state_tuple[0] == (0, 0):
                    break

                current_F = future_F

        fidelity_list.append(fidelity_count/step_count)
        step_count_list.append(step_count)


    plt.plot(range(len(step_count_list)), step_count_list, 'b-')
    plt.plot(range(len(step_count_list)), step_count_list, 'ro')
    # plt.show()
    plt.savefig('agent_step_history.png')

    plt.clf()

    plt.plot(range(len(fidelity_list)), fidelity_list, 'b-')
    plt.plot(range(len(fidelity_list)), fidelity_list, 'ro')
    plt.show()


def create_state_dict():
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


def create_action_dict():
    available_actions_list = list()
    i = 0
    for q1 in ((-1,), (1,),):
        for q2 in ((-1,), (1,),):
            for q3 in ((-1,), (1,),):
                if i < 5:
                    available_actions_list.append(q1 + q2 + q3)
                else:
                    break
                i += 1
            if i >= 5:
                break
        if i >= 5:
            break
    return available_actions_list


def create_reward_function_tuple():
    reward_function_tuple = (
        (220, 200, 180, 160, 140),
        (200, 180, 160, 120, 120),
        (180, 160, 0, 120, 100))
    return reward_function_tuple


def create_available_actions():
    available_actions_per_position_tuple = (
        ((1, 2, 4), (1, 2, 3, 4), (1, 3, 4), (1, 2, 3, 4), (2, 3, 4)),
        ((0, 1, 2, 4), (0, 2, 3, 4), tuple(), (0, 1, 2, 4), (0, 2, 3, 4)),
        ((0, 1, 4), (0, 1, 3, 4), (1, 3, 4), (0, 1, 3, 4), (0, 3, 4)))
    return available_actions_per_position_tuple


def create_optimal_policy_tuple():
    optimal_policy_tuple = (
        ((4,), (3,), (3,), (3,), (3,)),
        ((0,), (3, 0,), tuple(), (0,), (3, 0,)),
        ((0,), (3, 0,), (3,), (0,), (3, 0,)))
    return optimal_policy_tuple


def get_next_state(action, current_state):
    if action == 0:
        new_position = (
            current_state[0][0] - 1,
            current_state[0][1])

    elif action == 1:
        new_position = (
            current_state[0][0],
            current_state[0][1] + 1)

    elif action == 2:
        new_position = (
            current_state[0][0] + 1,
            current_state[0][1])

    elif action == 3:
        new_position = (
            current_state[0][0],
            current_state[0][1] - 1)

    else:
        new_position = current_state[0]

    return new_position

def print_agent(x_y_position):
    '''
    Prints the agent by postion.

    x_y_position
        Tuple of row and column indexes.
    '''
    for i in range(3):
        line_string = ''
        for j in range(5):
            if x_y_position == (i,j,):
                line_string += 'X'
            elif (i,j,) == (0,0):
                line_string += 'G'
            elif (i,j) == (1,2):
                line_string += 'W'
            else:
                line_string += 'O'
        print(line_string)
    print()


def print_results(f_value_list):

    for action_index in range(5):
        plt.plot(range(len(f_value_list)), tuple(map(lambda e: e[action_index], f_value_list)), label=action_index)

    plt.legend()
    plt.show()

if __name__ == '__main__':
    agent = make_dbm_agent(4, 3)
    #test_update_function(agent)
    test_rl(agent)
