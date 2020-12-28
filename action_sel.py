"""
#################################
# action selection function
#################################
"""

#########################################################
# import libraries
import random
import numpy as np
from copy import deepcopy
from statefromloc import getstateloc


#########################################################
# Function definition


def action_explore(x, y, action_list, size, state_list, index, region_size, x_reg, y_reg):
    """
    This function is the random exploration for the UAV and updates its current state and location based on the chosen
    random action
    :param x: -
    :param y: -
    :param action_list: The list of available actions
    :param size: -
    :param state_list: -
    :param index: -
    :param region_size: Size of each region
    :param x_reg: UAV's longitude (Regional)
    :param y_reg: UAV's latitude (Regional)
    :return: This function returns a random action, the new location and state for the UAV
    """
    left_action = deepcopy(action_list)
    random.seed()
    if x_reg == 0:
        left_action.remove(2)  # Remove Left
    if y_reg == 0:
        left_action.remove(1)  # Remove Down
    if x_reg == region_size - 1:
        left_action.remove(3)  # Remove Right
    if y_reg == region_size - 1:
        left_action.remove(0)  # Remove Up
    chosen_action = random.choice(left_action)

    # 0: Up, 1: down, 2: Left, 3: Right, 4: No movement
    if chosen_action == 0:  # Go Up
        x_new = x_reg
        y_new = y_reg + 1
    elif chosen_action == 1:  # Go Down
        x_new = x_reg
        y_new = y_reg - 1
    elif chosen_action == 2:  # Go Left
        x_new = x_reg - 1
        y_new = y_reg
    elif chosen_action == 3:  # Go Right
        x_new = x_reg + 1
        y_new = y_reg
    else:  # Stay at the same Location
        x_new = x_reg
        y_new = y_reg

    new_state = getstateloc(x_new, y_new, region_size)
    return chosen_action, x_new, y_new, new_state


def action_exploit(x, y, action_list, size, state_list, qval, region_size, x_reg, y_reg):
    """
    This function is greedy exploitation based on the Q table history of each drone. It chooses the best action based
    on the experienced Q values in the history based on the current state.
    :param x: -
    :param y: -
    :param action_list: The possible available action for the UAV
    :param size: -
    :param state_list: UAV's current state
    :param qval: Q value matrix for the chosen UAV
    :param region_size: The size of the regional grid
    :param x_reg: UAV's current longitude
    :param y_reg: UAV's current latitude
    :return: This function returns the chosen greedy action, updated location, and state of the drone.
    """
    left_action = deepcopy(action_list)
    # 0: Up, 1: down, 2: Left, 3: Right, 4: No Movement
    if x_reg == 0:
        left_action.remove(2)  # Remove Left
    if y_reg == 0:
        left_action.remove(1)  # Remove Down
    if x_reg == region_size - 1:
        left_action.remove(3)  # Remove Right
    if y_reg == region_size - 1:
        left_action.remove(0)  # Remove Up

    left_states_qval = []
    taken_actions = []

    for action in left_action:
        left_states_qval.append(qval[state_list, action])
        taken_actions.append(action)

    maxqval = max(left_states_qval)
    max_index_qval = int(np.argmax(left_states_qval))
    chosen_action_greedy = taken_actions[max_index_qval]
    # 0: Up, 1: down, 2: Left, 3: Right, 4: No Movement

    if chosen_action_greedy == 0:  # Go Up
        x_new = x_reg
        y_new = y_reg + 1
    elif chosen_action_greedy == 1:  # Go Down
        x_new = x_reg
        y_new = y_reg - 1
    elif chosen_action_greedy == 2:  # Go Left
        x_new = x_reg - 1
        y_new = y_reg
    elif chosen_action_greedy == 3:  # Go Right
        x_new = x_reg + 1
        y_new = y_reg
    else:  # Stay at the same Location
        x_new = x_reg
        y_new = y_reg

    new_state = getstateloc(x_new, y_new, region_size)
    return np.array(chosen_action_greedy), x_new, y_new, new_state
