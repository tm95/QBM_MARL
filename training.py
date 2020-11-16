from datetime import datetime
import numpy as np
import random

def train(env, agent, nb_episodes, nb_steps, logger):
    exp_time = datetime.now().strftime('%Y%m%d-%H-%M-%S')

    optimal_policy_tuple = ( \
        ((4,), (3,), (3,), (3,), (3,)), \
        ((0,), (3, 0,), tuple(), (0,), (3, 0,)), \
        ((0,), (3, 0,), (3,), (0,), (3, 0,)), \
        )

    reward_function_tuple = ( \
        (220, 200, 180, 160, 140), \
        (200, 180, 160, 120, 120), \
        (180, 160, 0, 120, 100), \
        )
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
    available_actions_per_position_tuple = ( \
        ((1, 2, 4), (1, 2, 3, 4), (1, 3, 4), (1, 2, 3, 4), (2, 3, 4)), \
        ((0, 1, 2, 4), (0, 2, 3, 4), tuple(), (0, 1, 2, 4), (0, 2, 3, 4)), \
        ((0, 1, 4), (0, 1, 3, 4), (1, 3, 4), (0, 1, 3, 4), (0, 3, 4)), \
        )
    '''
        Action-Set Description
            0 : ^
            1 : >
            2 : v
            3 : <
            4 : hold
    '''

    print("training started at {}".format(exp_time))

    def print_agent(x_y_position):
        '''
        Prints the agent by postion.
        x_y_position
            Tuple of row and column indexes.
        '''
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

    for training_episode in range(nb_episodes):
        steps = 0
        env.seed(seed=1234)  # Uncomment to randomize grid
        state = env.reset()
        #state = random.choice(tuple(available_state_dict.items()))
        all_done = False
        rewards = []
        beta = 1

        actions_count = 2300
        fidelity_acc_list = actions_count * [0]
        done = False

        # reinforcement learning loop
        while not all_done and steps < nb_steps:
            #env.render(mode='human', highlight=True)

            steps += 1
            action_list = []
            action, q, hh = agent.policy(state[0], beta)
            #action, q, hh = agent.policy(state[1], beta)
            #fidelity_acc_list[0] += 1 if action in optimal_policy_tuple[state[0]][state[1]] else 0
            action_list.append(action)

            next_state, reward, done, info = env.step(action_list)


            #if action == 0:
            #    new_position = (state[0][0] - 1, state[0][1])
            #elif action == 1:
            #    new_position = (state[0][0], state[0][1] + 1)
            #elif action == 2:
            #    new_position = (state[0][0] + 1, state[0][1])
            #elif action == 3:
            #    new_position = (state[0][0], state[0][1] - 1)
            #else:
            #    new_position = state[0]

            #if new_position in available_state_dict.keys():
            #    next_state = (new_position, available_state_dict[new_position])
            #else:
            #    next_state = state

            #reward = reward_function_tuple[next_state[0][0]][next_state[0][1]]

            #if state[0] == (0, 0):
            #    reward = reward_function_tuple[next_state[0][0]][next_state[0][1]]
            #    done = True
            #else:
            #    reward = -2.0

            #print_agent(state[0])
            #reward = reward[0]

            if reward == 0:
                reward = -10.0

            #print (reward)

            #reward = reward_function_tuple[next_state[0][0]][next_state[0][1]]

            agent.qlearn(state[0], action, reward, next_state[0], 0.01, q, hh)
            rewards.append(reward)
            state = next_state
            all_done = done

        if logger is not None:
            logger.log_metric('episode_return', training_episode, np.sum(rewards))
            logger.log_metric('episode_steps', training_episode, steps)

        for i in range(len(env.agents)):
            if logger is not None:
                logger.log_metric('episode_return_agent-{}'.format(i), training_episode, np.sum(rewards))

        print("episode {} finished at step {} with reward: {} at timestamp {}".format(
            training_episode, steps, np.sum(rewards), datetime.now().strftime('%Y%m%d-%H-%M-%S')))
