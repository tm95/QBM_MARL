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
from agent_utils import *
import multiprocessing
import os
import matplotlib.pyplot as plt

optimal_policy_tuple = (\
    ( (4,) , (3,)   , (3,)    , (3,) , (3,)   ) ,\
    ( (0,) , (3,0,) , tuple() , (0,) , (3,0,) ) ,\
    ( (0,) , (3,0,) , (3,)    , (0,) , (3,0,) ) ,\
)

reward_function_tuple = (\
    ( 220 , 200 , 180 , 160 , 140 ) ,\
    ( 200 , 180 , 160 , 120 , 120 ) ,\
    ( 180 , 160 ,   0 , 120 , 100 ) ,\
)
available_state_dict = dict()
i = 0
for q1 in ( ( -1 , ) , ( 1 , ) ,):
    for q2 in ( ( -1 , ) , ( 1 , ) ,):
        for q3 in ( ( -1 , ) , ( 1 , ) ,):
            for q4 in ( ( -1 , ) , ( 1 , ) ,):
                if i != 7:
                    available_state_dict[ (i//5,i%5,) ] = q1 + q2 + q3 + q4
                i += 1
                if i >= 15: break
            if i >= 15: break
        if i >= 15: break
    if i >= 15: break
available_actions_list = list()
i = 0
for q1 in ( ( -1 , ) , ( 1 , ) ,):
    for q2 in ( ( -1 , ) , ( 1 , ) ,):
        for q3 in ( ( -1 , ) , ( 1 , ) ,):
            if i < 5:
                available_actions_list.append(q1 + q2 + q3)
            else:
                break
            i+=1
        if i >= 5:
            break
    if i >= 5:
        break
available_actions_per_position_tuple = (\
    ( (1,2,4)   , (1,2,3,4) , (1,3,4) , (1,2,3,4) , (2,3,4)   ) ,\
    ( (0,1,2,4) , (0,2,3,4) , tuple() , (0,1,2,4) , (0,2,3,4) ) ,\
    ( (0,1,4)   , (0,1,3,4) , (1,3,4) , (0,1,3,4) , (0,3,4)   ) ,\
)
'''
    Action-Set Description
        0 : ^
        1 : >
        2 : v
        3 : <
        4 : hold
'''


# def agent_step(current_state):
def agent_step(current_state, Q_hh, Q_vh, epsilon_p):
    '''
    Implements a Grid World problem step.
    '''
    max_tuple = None

    if not ( 0 <= current_state[0][0] < 3 and 0 <= current_state[0][1] ):
        print('first debug:', current_state)

    if epsilon_p == 0 or random.random() > epsilon_p:

        # actions_energies_list = list()

        for action_index in filter(
                # lambda e: e != 4,
                lambda e: True,
                available_actions_per_position_tuple[current_state[0][0]][current_state[0][1]]):

            vis_iterable = current_state[1] + available_actions_list[action_index]

            general_Q = create_general_Q_from(Q_hh, Q_vh, vis_iterable)

            samples = list(SimulatedAnnealingSampler().sample_qubo(general_Q, num_reads=sample_count, vartype=0).samples())

            random.shuffle(samples)

            current_F = get_free_energy(get_3d_hamiltonian_average_value(samples, general_Q, replica_count, average_size, 0.5, 2),
                                        samples, replica_count, 2)

            if max_tuple is None or max_tuple[0] < current_F:
                max_tuple = (current_F, action_index, samples, vis_iterable)

    else:

        action_index = random.choice(
            tuple(filter(
                # lambda e: e != 4,
                lambda e: True,
                available_actions_per_position_tuple[\
                current_state[0][0]][current_state[0][1]]
            )))

        vis_iterable = current_state[1] + available_actions_list[action_index]

        general_Q = create_general_Q_from(Q_hh, Q_vh, vis_iterable)

        samples = list(SimulatedAnnealingSampler().sample_qubo(
            general_Q,
            num_reads=sample_count, vartype=0
        ).samples())

        random.shuffle(samples)

        current_F = get_free_energy(
            get_3d_hamiltonian_average_value(
                samples,
                general_Q,
                replica_count,
                average_size,
                0.5,
                2
            ),
            samples,
            replica_count,
            2,
        )

        max_tuple = ( current_F , action_index , samples , vis_iterable )

    if max_tuple[1] == 0:
        new_position = (\
            current_state[0][0] - 1,
            current_state[0][1]
        )
    elif max_tuple[1] == 1:
        new_position = (\
            current_state[0][0],
            current_state[0][1] + 1,
        )
    elif max_tuple[1] == 2:
        new_position = (\
            current_state[0][0] + 1,
            current_state[0][1],
        )
    elif max_tuple[1] == 3:
        new_position = (\
            current_state[0][0],
            current_state[0][1] - 1,
        )
    else:
        new_position = current_state[0]


    if not ( 0 <= new_position[0] < 3 and 0 <= new_position[1] ):
        print('second debug:',new_position, max_tuple)

    return\
        (
            (
                new_position,
                available_state_dict[new_position],
            ),
            max_tuple[0],     # F value
            max_tuple[2],     # samples
            max_tuple[3],     # used iterable vector
            max_tuple[1],     # action index
            current_state[0], # old_position
        )

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


def paper_simulation_main():
    '''
    Single core multiple independent runs if the Grid World problem agent.
    '''

    global Q_hh, Q_vh, sample_count, replica_count, average_size


    Q_hh = dict()
    for i, ii in zip(tuple(range(4)), tuple(range(8, 12))):
        for j, jj in zip(tuple(range(4, 8)), tuple(range(12, 16))):
            Q_hh[(i, j)] = 2 * random.random() - 1
            Q_hh[(ii, jj)] = 2 * random.random() - 1
    for i, j in zip(tuple(range(4, 8)), tuple(range(12, 16))):
        Q_hh[(i, j)] = 2 * random.random() - 1

    Q_vh = dict()
    # Fully connection between state and blue nodes
    for j in (tuple(range(4)) + tuple(range(12, 16))):
        for i in range(4):
            Q_vh[(i, j,)] = 2 * random.random() - 1
        # Fully connection between action and red nodes
    for j in (tuple(range(4, 8)) + tuple(range(8, 12))):
        for i in range(4, 7):
            Q_vh[(i, j,)] = 2 * random.random() - 1

    runs_count = 500
    actions_count = 2300

    replica_count = 20
    average_size = 50
    sample_count = replica_count * average_size
    epsilon = 1
    steps = 200

    for run_index in range(runs_count):
        sample_index = 1
        done = False
        fidelity_acc_list = actions_count * [0]

        agent_state_tuple = random.choice(tuple(available_state_dict.items()))

        agent_state_tuple, current_F, current_samples, current_vis_iterable, action_index, old_position_tuple = agent_step(agent_state_tuple, Q_hh, Q_vh, 0.3)
        fidelity_acc_list[0] += 1 if action_index in optimal_policy_tuple[old_position_tuple[0]][old_position_tuple[1]] else 0

        while sample_index < steps and not done:
            sample_index += 1
            epsilon -= 0.01
            agent_state_tuple, future_F, future_samples, future_vis_iterable, action_index, old_position_tuple = \
                agent_step(agent_state_tuple, Q_hh, Q_vh, epsilon)
            fidelity_acc_list[sample_index] += 1 if action_index in optimal_policy_tuple[old_position_tuple[0]][old_position_tuple[1]] else 0

            if agent_state_tuple[0] == (0,0):
                done = True

            #print_agent(old_position_tuple)
            # sleep(1)

            Q_hh, Q_vh = update_weights(Q_hh, Q_vh, current_samples, reward_function_tuple[agent_state_tuple[0][0]][agent_state_tuple[0][1]], 0, current_F,
                                        current_vis_iterable, 0.001, 0.8)

            current_F, current_samples, current_vis_iterable = future_F, future_samples, future_vis_iterable

        print (sum(fidelity_acc_list)/sample_index)

def test_main_0():
        '''
        Tests the updating mechanism. The free energies should go
        towards the values in "reward_tuple".
        '''
        global replica_count, \
            average_size, \
            sample_count

        Q_hh = dict()
        for i, ii in zip(tuple(range(4)), tuple(range(8, 12))):
            for j, jj in zip(tuple(range(4, 8)), tuple(range(12, 16))):
                Q_hh[(i, j)] = 2 * random.random() - 1
                Q_hh[(ii, jj)] = 2 * random.random() - 1
        for i, j in zip(tuple(range(4, 8)), tuple(range(12, 16))):
            Q_hh[(i, j)] = 2 * random.random() - 1

        Q_vh = dict()
        # Fully connection between state and blue nodes
        for j in (tuple(range(4)) + tuple(range(12, 16))):
            for i in range(4):
                Q_vh[(i, j,)] = 2 * random.random() - 1
            # Fully connection between action and red nodes
        for j in (tuple(range(4, 8)) + tuple(range(8, 12))):
            for i in range(4, 7):
                Q_vh[(i, j,)] = 2 * random.random() - 1

        replica_count = 3
        average_size = 5
        sample_count = replica_count * average_size

        reward_tuple = \
            ( \
                500, \
                400, \
                300, \
                200, \
                100, \
                )

        tries_count = 3000

        agent_state_tuple = random.choice(tuple(available_state_dict.items()))

        f_value_list = list()

        for _ in range(tries_count):

            f_value_list.append(list())

            for action_index in range(5):
                vis_iterable = agent_state_tuple[1] + available_actions_list[action_index]

                general_Q = create_general_Q_from(
                    Q_hh,
                    Q_vh,
                    vis_iterable
                )

                samples = list(SimulatedAnnealingSampler().sample_qubo(
                    general_Q,
                    num_reads=sample_count, vartype=0
                ).samples())

                random.shuffle(samples)

                current_F = get_free_energy(
                    get_3d_hamiltonian_average_value(
                        samples,
                        general_Q,
                        replica_count,
                        average_size,
                        0.5,
                        2
                    ),
                    samples,
                    replica_count,
                    2,
                )

                Q_hh, Q_vh = \
                    update_weights(
                        Q_hh,
                        Q_vh,
                        samples,
                        reward_tuple[action_index],
                        0,
                        current_F,
                        vis_iterable,
                        0.0001,
                        0.8
                    )

                f_value_list[-1].append(current_F)

            # print( f_value_list[-1] )

        plt.plot(
            range(len(f_value_list)),
            tuple(map(lambda e: e[0], f_value_list)),
            label='First'
        )
        plt.plot(
            range(len(f_value_list)),
            tuple(map(lambda e: e[1], f_value_list)),
            label='Second'
        )
        plt.plot(
            range(len(f_value_list)),
            tuple(map(lambda e: e[2], f_value_list)),
            label='Third'
        )
        plt.plot(
            range(len(f_value_list)),
            tuple(map(lambda e: e[3], f_value_list)),
            label='Fourth'
        )
        plt.plot(
            range(len(f_value_list)),
            tuple(map(lambda e: e[4], f_value_list)),
            label='Fifth'
        )
        plt.legend()
        plt.show()

if __name__ == '__main__':

    #paper_simulation_main()
    test_main_0()
