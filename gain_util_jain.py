"""
#################################
# Reward function for Multi Agent Q learning
#################################
"""

#########################################################
# import libraries
import numpy as np

#########################################################
# Function definition


def reward_val_single(u_net, u_net_arr, u_sum, u_sum_arr, energy, energy_prev, mobility_rate):
    """
    This function calculates reward for all UAVs based on the current and previous throughput values and considering
    the energy consumption as well. The rewards are constant values.
    :param u_net: Throughput of all UAVs at the current step
    :param u_net_arr: Throughput of all UAVs from the beginning
    :param u_sum: -
    :param u_sum_arr: -
    :param energy: Remaining energy of all UAVs
    :param energy_prev: Previous step remaining energy
    :param mobility_rate: Energy consumption rate
    :return: This function returns all reward values for all UAVs
    """
    if u_net_arr.size == u_net.size:
        max_u_net = np.zeros([u_net.size], dtype=float)
    else:
        max_u_net = np.max(u_net_arr, axis=0)

    reward = np.zeros([u_net.size, 1], dtype=float)

    #  TODO: Checked!: Consider the energy value in the reward function too!
    #  Reward behavior: 1
    #  *******************************************************
    for uav in np.arange(u_net.size):
        if (u_net[uav] > max_u_net[uav]) and (energy_prev[uav] - energy[uav] > mobility_rate):  # Very Good
            reward[uav] = 10
        if (u_net[uav] - max_u_net[uav] == 0) and (energy_prev[uav] - energy[uav] < mobility_rate):  # Good1
            reward[uav] = 2
        if (u_net[uav] - max_u_net[uav] == 0) and (energy_prev[uav] - energy[uav] > mobility_rate):  # Good2
            reward[uav] = 4
        if (u_net[uav] - max_u_net[uav] < 0) and (energy_prev[uav] - energy[uav] < mobility_rate):  # Bad
            reward[uav] = -0.2
        if (u_net[uav] - max_u_net[uav] < 0) and (energy_prev[uav] - energy[uav] > mobility_rate):  # Very Bad
            reward[uav] = -0.5
    #  *******************************************************
    return np.squeeze(reward)
