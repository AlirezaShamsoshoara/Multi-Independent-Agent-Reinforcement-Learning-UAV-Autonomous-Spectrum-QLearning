"""
#################################
# Ground Station function for Controlling the UAVs
#################################
"""

#########################################################
# import libraries
import numpy as np
from copy import deepcopy
from scipy.stats import rankdata
from statefromloc import getstateloc
import scipy.spatial.distance as ssd


#########################################################
# Function definition


def regioncenter(size, region, height):
    """
    This function calculates the center coordinates of each region.
    :param size: The global grid size
    :param region: Number of regions in the global grid
    :param height: The altitude of all UAVs
    :return: This function returns the regions' center coordinates along with their IDs
    """
    dim = np.sqrt(region)
    dim = int(dim)
    length = size / dim
    length = int(length)

    reg_coordinates = []

    for j in range(0, dim):
        for i in range(0, dim):
            x = int(i * length + length/2)
            y = int(j * length + length/2)
            reg_coordinates.append((x, y, height))

    reg_coordinates_arr = np.asarray(reg_coordinates)
    reg_id = getstateloc(reg_coordinates_arr[:, 0], reg_coordinates_arr[:, 1], size)
    return reg_coordinates_arr, reg_id


def search_primary_region(regions, locations, region_ids):
    """
    This function finds the best region for the primary UAV to relay the terrestrial users' data.
    :param regions: regions coordinates (centers)
    :param locations: Location dictionary includes all location
    :param region_ids: IDs of regions
    :return: returns the value, index, and the ID of the primary region
    """
    x_p = locations.get('X_GR')
    y_p = locations.get('Y_GR')
    z_p = locations.get('Z_GR')

    dist_regions = [ssd.euclidean([x_p, y_p, z_p], [i, j, k]) for i, j, k in regions]
    min_reg_value = np.amin(dist_regions)
    min_reg_idx = np.argmin(dist_regions)
    primary_region_id = region_ids[min_reg_idx]
    return min_reg_value, min_reg_idx, primary_region_id


def search_primary_uav(regions, min_reg_idx, pu_reg_id, locations, num_uav, energy, mob_energy_rate):
    """
    This function finds the best UAV for the primary network after its flight considering the residual energy.
    :param regions: Coordinates (centers) of regions
    :param min_reg_idx: Index of the primary region
    :param pu_reg_id: -
    :param locations: Location dictionary includes all location
    :param num_uav: Number of UAVs
    :param energy: Energy matrix of all drones
    :param mob_energy_rate: The energy consumption rate for mobility
    :return: This function returns index and ID of the best optimal UAV with the residual energy after flight.
    """
    x_u = locations.get('X_U')
    y_u = locations.get('Y_U')
    z_u = locations.get('Z_U')

    dist_uav_p = [ssd.euclidean(regions[min_reg_idx, :], [i, j, k]) for i, j, k in zip(x_u, y_u, z_u)]
    dist_uav_p = np.asarray(dist_uav_p)
    nearest_uav_pu_idx = np.argmin(dist_uav_p)
    flight_distance = np.zeros(num_uav, dtype=int)
    dist_vector = []

    for uav in range(0, num_uav):
        dist = np.abs(regions[min_reg_idx, :] - np.squeeze(np.asarray([x_u[uav], y_u[uav], z_u[uav]])))
        dist_vector.append(dist)
        flight_distance[uav] = np.sum(dist)

    residual_energy_after_flight = [energy[uav] - mob_energy_rate * flight_distance[uav] for uav in range(0, num_uav)]
    residual_energy_after_flight = np.asarray(residual_energy_after_flight)
    optimal_uav_energy_idx = np.argmax(residual_energy_after_flight)

    return nearest_uav_pu_idx, optimal_uav_energy_idx, residual_energy_after_flight


def search_secondary_uav(num_uav, pu_reg, pu_reg_id, pu_uav_id, regions, energy, locations, mob_energy_rate, reg_ids):
    """
    This function allocates secondary UAVs to the secondary regions based on the best allocation of residual energy.
    :param num_uav: Number of UAVs
    :param pu_reg: Region of the primary (location)
    :param pu_reg_id: Region of the primary (ID)
    :param pu_uav_id: ID of the primary UAV
    :param regions: Coordinates (centers) of regions
    :param energy: Energy matrix of all drones
    :param locations: Location dictionary includes all location
    :param mob_energy_rate: The energy consumption rate for mobility
    :param reg_ids: -
    :return: This function returns residual energy after flight of all UAVs with the preference vector of UAVs and
    the sorted IDs of UAVs with the allocated regions.
    """
    x_u = locations.get('X_U')
    y_u = locations.get('Y_U')
    z_u = locations.get('Z_U')

    x_f = locations.get('X_F')
    y_f = locations.get('Y_F')
    z_f = locations.get('Z_F')

    dist_regions = [ssd.euclidean([x_f, y_f, z_f], [i, j, k]) for i, j, k in regions]
    dist_regions = np.asarray(dist_regions)
    dist_regions_sort = np.sort(dist_regions)
    dist_regions_sort_idx = np.argsort(dist_regions)

    index = np.argwhere(dist_regions_sort_idx == pu_reg)
    dist_regions_sort_idx_su = np.delete(dist_regions_sort_idx, index)

    picked_regions_su = dist_regions_sort_idx_su[0: num_uav - 1]

    dist_matrix = np.zeros([num_uav, num_uav-1], dtype=float)
    move_matrix = np.zeros([num_uav, num_uav-1], dtype=int)
    energy_matrix = np.zeros([num_uav, num_uav - 1], dtype=float)
    preference_matrix = np.zeros([num_uav, num_uav - 1], dtype=int)
    preference_array = np.zeros(num_uav, dtype=int)
    preference_array[pu_uav_id] = -1
    for uav in range(0, num_uav):
        if uav == pu_uav_id:
            continue
        index = 0
        for region in picked_regions_su:
            dist_matrix[uav, index] = ssd.euclidean([x_u[uav], y_u[uav], z_u[uav]], [regions[region, 0],
                                                                                     regions[region, 1],
                                                                                     regions[region, 2]])
            move_matrix[uav, index] = np.sum(np.abs(regions[region, :] - np.squeeze(np.asarray([x_u[uav], y_u[uav],
                                                                                                z_u[uav]]))))
            energy_matrix[uav, index] = energy[uav] - mob_energy_rate * move_matrix[uav, index]
            index = index + 1
        preference_matrix[uav, :] = np.asarray(rankdata(energy_matrix[uav, :], method='ordinal'), dtype=int)
        preference_array[uav] = np.argmax(preference_matrix[uav, :])

        check = 1
        while True:
            if preference_array[uav] in preference_array[0:uav]:
                # preference_array[uav] = np.asscalar(np.where(preference_matrix[uav, :] == num_uav-1-check)[0])
                preference_array[uav] = np.where(preference_matrix[uav, :] == num_uav-1-check)[0].item()
            else:
                break
            check = check + 1
    preference_array[pu_uav_id] = -1

    preference_array_reg_id = picked_regions_su[preference_array]
    preference_array_reg_id[pu_uav_id] = pu_reg
    energy_matrix[pu_uav_id, :] = energy[pu_uav_id]

    flight_distance = np.zeros(num_uav, dtype=int)
    dist_vector = []

    for uav in range(0, num_uav):
        dist = np.abs(regions[preference_array_reg_id[uav], :] - np.squeeze(np.asarray([x_u[uav], y_u[uav], z_u[uav]])))
        dist_vector.append(dist)
        flight_distance[uav] = np.sum(dist)

        x_u[uav], y_u[uav], z_u[uav] = regions[preference_array_reg_id[uav], :]

    residual_energy_after_flight = [energy[uav] - mob_energy_rate * flight_distance[uav] for uav in range(0, num_uav)]
    residual_energy_after_flight = np.asarray(residual_energy_after_flight)

    su_uav_id = np.asarray(range(0, num_uav))
    su_uav_id = np.delete(su_uav_id, pu_uav_id)

    su_regions_id = np.delete(preference_array_reg_id, pu_uav_id)

    return residual_energy_after_flight, preference_array_reg_id, su_uav_id, su_regions_id


# ********************************* Module for base station: random UAV allocation
def random_primary_uav(regions, min_reg_idx, locations, num_uav, energy, mob_energy_rate):
    """
    This function randomly chooses a UAV for the primary region and flies it and updates its residual energy.
    :param regions: Coordinates (centers) of regions
    :param min_reg_idx: Index of the primary region
    :param locations: Location dictionary includes all location
    :param num_uav: Number of UAVs
    :param energy: Energy matrix of all drones
    :param mob_energy_rate: The energy consumption rate for mobility
    :return: This function returns a random UAV ID allocated to the primary regions with the updated energy matrix.
    """
    x_u = locations.get('X_U')
    y_u = locations.get('Y_U')
    z_u = locations.get('Z_U')

    rand_pu_uav = np.random.randint(0, num_uav)
    dist = np.abs(regions[min_reg_idx, :] - np.squeeze(np.asarray([x_u[rand_pu_uav], y_u[rand_pu_uav],
                                                                   z_u[rand_pu_uav]])))
    flight_distance = np.sum(dist)
    residual_energy_after_flight = deepcopy(energy)
    residual_energy_after_flight[rand_pu_uav] = energy[rand_pu_uav] - mob_energy_rate * flight_distance
    return rand_pu_uav, residual_energy_after_flight


def random_secondary_uav_search_region(num_uav, pu_reg, pu_uav_id, regions, energy, locations, mob_energy_rate):
    """
    This function randomly allocates UAVs to the secondary regions and updates their
    energy after flight.
    :param num_uav: Number of UAVs
    :param pu_reg: Region of the primary (location)
    :param pu_uav_id: Region of the primary (ID)
    :param regions: Coordinates (centers) of regions
    :param energy: Energy matrix of all drones
    :param locations: Location dictionary includes all location
    :param mob_energy_rate: The energy consumption rate for mobility
    :return: This function returns residual energy after flight of all UAVs with the preference vector of UAVs and
    the sorted IDs of UAVs with the allocated regions.
    """
    x_u = locations.get('X_U')
    y_u = locations.get('Y_U')
    z_u = locations.get('Z_U')

    x_f = locations.get('X_F')
    y_f = locations.get('Y_F')
    z_f = locations.get('Z_F')

    dist_regions = [ssd.euclidean([x_f, y_f, z_f], [i, j, k]) for i, j, k in regions]
    dist_regions = np.asarray(dist_regions)
    dist_regions_sort_idx = np.argsort(dist_regions)
    index = np.argwhere(dist_regions_sort_idx == pu_reg)
    dist_regions_sort_idx_su = np.delete(dist_regions_sort_idx, index)
    picked_regions_su = dist_regions_sort_idx_su[0: num_uav - 1]

    su_uav_id = np.arange(num_uav)
    su_uav_id = np.delete(su_uav_id, pu_uav_id, 0)
    preference_array_su = np.arange(num_uav-1)
    np.random.shuffle(preference_array_su)
    preference_array = np.zeros([num_uav], dtype=int)
    preference_array[su_uav_id] = preference_array_su
    preference_array[pu_uav_id] = -1

    preference_array_reg_id = picked_regions_su[preference_array]
    preference_array_reg_id[pu_uav_id] = pu_reg

    flight_distance = np.zeros(num_uav, dtype=int)
    dist_vector = []

    for uav in range(0, num_uav):
        dist = np.abs(regions[preference_array_reg_id[uav], :] - np.squeeze(np.asarray([x_u[uav], y_u[uav], z_u[uav]])))
        dist_vector.append(dist)
        flight_distance[uav] = np.sum(dist)

        x_u[uav], y_u[uav], z_u[uav] = regions[preference_array_reg_id[uav], :]

    residual_energy_after_flight = [energy[uav] - mob_energy_rate * flight_distance[uav] for uav in range(0, num_uav)]
    residual_energy_after_flight = np.asarray(residual_energy_after_flight)
    su_regions_id = np.delete(preference_array_reg_id, pu_uav_id)

    return residual_energy_after_flight, preference_array_reg_id, su_uav_id, su_regions_id


# ********************************* Module for base station: random region allocation for primary and secondary and
# random secondary uav selection
def random_primary_region(region_ids):
    """
    This function chooses a random region for the primary UAV
    :param region_ids: IDs of regions
    :return: random region for the primary (index, ID)
    """
    primary_region_id = np.random.choice(region_ids)
    min_reg_idx = np.where(primary_region_id == region_ids)[0].item()
    return min_reg_idx, primary_region_id


def random_secondary_uav_random_region(num_uav, pu_reg, pu_uav_id, regions, energy, locations, mob_energy_rate,
                                       reg_ids):
    """
    This function randomly chooses regions for the secondary network and randomly allocates the UAVs to those regions.
    :param num_uav: Number of UAVs
    :param pu_reg: Region of the primary (location)
    :param pu_uav_id: ID of the primary UAV
    :param regions: Coordinates (centers) of regions
    :param energy: Energy matrix of all drones
    :param locations: Location dictionary includes all location
    :param mob_energy_rate: The energy consumption rate for mobility
    :param reg_ids: IDs of regions
    :return: This function returns residual energy after flight of all UAVs with the preference vector of UAVs and
    the sorted IDs of UAVs with the allocated regions.
    """
    x_u = locations.get('X_U')
    y_u = locations.get('Y_U')
    z_u = locations.get('Z_U')

    su_regions = np.arange(reg_ids.size)
    su_regions = np.delete(su_regions, pu_reg)
    picked_regions_su = np.random.choice(su_regions, num_uav - 1, replace=False)

    su_uav_id = np.arange(num_uav)
    su_uav_id = np.delete(su_uav_id, pu_uav_id, 0)
    preference_array_su = np.arange(num_uav - 1)
    np.random.shuffle(preference_array_su)
    preference_array = np.zeros([num_uav], dtype=int)
    preference_array[su_uav_id] = preference_array_su
    preference_array[pu_uav_id] = -1

    preference_array_reg_id = picked_regions_su[preference_array]
    preference_array_reg_id[pu_uav_id] = pu_reg

    flight_distance = np.zeros(num_uav, dtype=int)
    dist_vector = []

    for uav in range(0, num_uav):
        dist = np.abs(regions[preference_array_reg_id[uav], :] - np.squeeze(np.asarray([x_u[uav], y_u[uav], z_u[uav]])))
        dist_vector.append(dist)
        flight_distance[uav] = np.sum(dist)

        x_u[uav], y_u[uav], z_u[uav] = regions[preference_array_reg_id[uav], :]

    residual_energy_after_flight = [energy[uav] - mob_energy_rate * flight_distance[uav] for uav in range(0, num_uav)]
    residual_energy_after_flight = np.asarray(residual_energy_after_flight)
    su_regions_id = picked_regions_su

    return residual_energy_after_flight, preference_array_reg_id, su_uav_id, su_regions_id


# ********************************* Module for base station random region allocation for primary and secondary and
# searching algorithm to find the best secondary uavs
def search_secondary_uav_random_region(num_uav, pu_reg, pu_uav_id, regions, energy, locations, mob_energy_rate,
                                       reg_ids):
    """
    This function randomly chooses regions for the secondary network but allocates UAVs to them based on the best
    allocation.
    :param num_uav: Number of UAVs
    :param pu_reg: Region of the primary (location)
    :param pu_uav_id: ID of the primary UAV
    :param regions: Coordinates (centers) of regions
    :param energy: Energy matrix of all drones
    :param locations: Location dictionary includes all location
    :param mob_energy_rate: The energy consumption rate for mobility
    :param reg_ids: regions IDs
    :return: This function returns residual energy after flight of all UAVs with the preference vector of UAVs and
    the sorted IDs of UAVs with the allocated regions.
    """
    x_u = locations.get('X_U')
    y_u = locations.get('Y_U')
    z_u = locations.get('Z_U')

    su_regions = np.arange(reg_ids.size)
    su_regions = np.delete(su_regions, pu_reg)
    picked_regions_su = np.random.choice(su_regions, num_uav - 1, replace=False)

    move_matrix = np.zeros([num_uav, num_uav - 1], dtype=int)
    energy_matrix = np.zeros([num_uav, num_uav - 1], dtype=float)
    preference_matrix = np.zeros([num_uav, num_uav - 1], dtype=int)
    preference_array = np.zeros(num_uav, dtype=int)
    preference_array[pu_uav_id] = -1
    for uav in range(0, num_uav):
        if uav == pu_uav_id:
            continue
        index = 0
        for region in picked_regions_su:
            move_matrix[uav, index] = np.sum(np.abs(regions[region, :] - np.squeeze(np.asarray([x_u[uav], y_u[uav],
                                                                                                z_u[uav]]))))
            energy_matrix[uav, index] = energy[uav] - mob_energy_rate * move_matrix[uav, index]
            index = index + 1
        preference_matrix[uav, :] = np.asarray(rankdata(energy_matrix[uav, :], method='ordinal'), dtype=int)
        preference_array[uav] = np.argmax(preference_matrix[uav, :])

        check = 1
        while True:
            if preference_array[uav] in preference_array[0:uav]:
                preference_array[uav] = np.where(preference_matrix[uav, :] == num_uav - 1 - check)[0].item()
            else:
                break
            check = check + 1
    preference_array[pu_uav_id] = -1

    preference_array_reg_id = picked_regions_su[preference_array]
    preference_array_reg_id[pu_uav_id] = pu_reg
    energy_matrix[pu_uav_id, :] = energy[pu_uav_id]

    flight_distance = np.zeros(num_uav, dtype=int)
    dist_vector = []

    for uav in range(0, num_uav):
        dist = np.abs(regions[preference_array_reg_id[uav], :] - np.squeeze(np.asarray([x_u[uav], y_u[uav], z_u[uav]])))
        dist_vector.append(dist)
        flight_distance[uav] = np.sum(dist)

        x_u[uav], y_u[uav], z_u[uav] = regions[preference_array_reg_id[uav], :]

    residual_energy_after_flight = [energy[uav] - mob_energy_rate * flight_distance[uav] for uav in range(0, num_uav)]
    residual_energy_after_flight = np.asarray(residual_energy_after_flight)

    su_uav_id = np.asarray(range(0, num_uav))
    su_uav_id = np.delete(su_uav_id, pu_uav_id)

    su_regions_id = np.delete(preference_array_reg_id, pu_uav_id)

    return residual_energy_after_flight, preference_array_reg_id, su_uav_id, su_regions_id
