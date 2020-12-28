""""
#################################
# Energy function
#################################
"""

#########################################################
# import libraries
import numpy as np
import scipy.io as sio

#########################################################
# Function definition


def init_energy(num_UAV, min_energy, max_energy, savefile, pthenergy):
    """
    This function initializes the energy values for all drones and UAV randomly based on the energy config file
    :param num_UAV: Number of UAVs
    :param min_energy: Minimum possible value for the initialization
    :param max_energy: Maximum possible value for the initialization
    :param savefile: A FLAG to Save or Not Save the energy values on disk
    :param pthenergy: The path to save the energy values
    :return: This function returns the initial energy values
    """
    if savefile:
        energy = np.random.uniform(min_energy, max_energy, num_UAV)
        energy_dict = dict([('energy', energy)])
        sio.savemat(pthenergy, energy_dict)
    else:
        energy_dict = sio.loadmat(pthenergy)
        energy = energy_dict.get('energy')
        energy = np.squeeze(energy)
    return energy


def update_energy_movement(energy, actions, mobility_rate):
    """
    This function updates the UAVs' energy after choosing an action for the mobility.
    :param energy: The energy matrix(vector) for all UAVs
    :param actions: Chosen action from Q learning
    :param mobility_rate: The energy consumption rate for mobility
    :return: This function returns the updated energy values after the chosen action
    """
    num_uav = actions.size
    no_movement = 4
    returned_energy = np. zeros([num_uav], dtype=float)
    for uav in np.arange(num_uav):
        returned_energy[uav] = energy[uav] - mobility_rate if actions[uav] < no_movement else energy[uav]
    return returned_energy


def update_energy_transmission(energy, tran_rate):
    """
    This function updates the UAVs' energy after data transmission.
    :param energy: The energy matrix(vector) for all UAVs
    :param tran_rate: The energy consumption rate for transmission
    :return: This function returns the updated energy values after each transmission
    """
    num_uav = energy.size
    returned_energy = np.zeros([num_uav], dtype=float)
    for uav in np.arange(num_uav):
        returned_energy[uav] = energy[uav] - tran_rate if energy[uav] > 0 else 0
    return returned_energy

