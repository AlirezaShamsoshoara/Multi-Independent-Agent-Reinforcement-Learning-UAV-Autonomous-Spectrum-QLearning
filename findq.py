"""
#################################
# Find Qmax for the Next State
#################################
"""

#########################################################
# import libraries
from copy import deepcopy

#########################################################
# Function definition


def find_max_q_next_single(qval, next_state, x_reg, y_reg, action_list, region_size):
    """
    This function finds the maximum Q values for the UAV based on the new state that it moved to.
    :param qval: Q value matrix for the UAV
    :param next_state: The new state that the UAV moved into it based on the chosen action
    :param x_reg: The new longitude that the UAV currently has
    :param y_reg: The new latitude that the UAV currently has
    :param action_list: The available and possible action list
    :param region_size: The size of the regional grid
    :return: This function returns the maximum Q value for updating the Q table in the main file
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
        left_states_qval.append(qval[next_state, action])
        taken_actions.append(action)

    maxqval = max(left_states_qval)
    return maxqval
