"""
#################################
# Get State from location function
#################################
"""

#########################################################
# import libraries
import numpy as np

#########################################################
# Function definition


def getstateloc(x_u, y_u, size):
    """
    This function returns the state based on the UAV's location.
    :param x_u: UAV's longitude
    :param y_u: UAV's latitude
    :param size: The grid size
    :return: The current state based on location
    """
    state = y_u * size + x_u
    return state


def getlocstate(state, size):
    """
    This function returns the UAV's current location based on the current state inside the region.
    :param state: UAV's current state
    :param size: The grid size
    :return: This function returns the UAV's location based on its current state.
    """
    x = np.mod(state, size)
    y = state / size
    return x, y


def get_region_state_from_general(general_state, general_size, region_size):
    """
    This function converts the UAV's state from the big and global grid to the smaller region also it returns the
    regional location as well.
    :param general_state: The global state inside the bigger grid
    :param general_size: The size of global grid
    :param region_size: Size of smaller regions
    :return: This function returns regional state and location for all UAVs
    """
    general_loc_x, general_loc_y = getlocstate(general_state, general_size)
    region_loc_x = np.mod(general_loc_x, region_size)
    region_loc_y = np.mod(general_loc_y, region_size)
    region_state = getstateloc(region_loc_x, region_loc_y, region_size)
    return region_state, np.squeeze(region_loc_x), np.squeeze(region_loc_y)


def get_general_loc_from_region(x_reg, y_reg, reg_id, region_size, general_size):
    """
    This function is the reverse of the previous function which means it converts the regional location to the global
    location.
    :param x_reg: regional UAV's longitude
    :param y_reg: regional UAV's latitude
    :param reg_id: ID of that specific region
    :param region_size: The size of the region
    :param general_size: The size of the global grid
    :return: This function returns the global location for each drone
    """
    x_gen = np.mod(reg_id, int(general_size/region_size)) * region_size + x_reg
    y_gen = int(reg_id/(int(general_size/region_size))) * region_size + y_reg
    return x_gen, y_gen
