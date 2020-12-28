"""
#################################
# General Utility for each UAV
#################################
"""

#########################################################
# import libraries
import numpy as np

#########################################################
# Function definition


def singleutil(action, csi, general, power):
    """
    This function calculates the Amplify and forward throughput rate using the SNR for all UAVs and return all values.
    :param action: This is task vector for all UAVs (Emergency or Relaying)
    :param csi: Channel State information between UAVs, sources, and destinations
    :param general: General parameters and configs
    :param power: The power transmission and consumption configuration
    :return: This function returns the calculated throughput rate for all UAVs
    """
    # source_uav = 0
    # uav_fusion = 1
    # gtuser_uav = 2
    # uav_gruser = 3

    h_s_uav = csi[:, 0]
    h_uav_f = csi[:, 1]
    h_gt_uav = csi[:, 2]
    h_uav_gr = csi[:, 3]
    total = np.zeros([action.size])
    if general.get('DF'):  # Here we're calculating the Decode and Forward (DF) Rate for the cooperation
        pass
    else:  # Here we're calculating the Amplify and Forward (AF) Rate for the cooperation
        # uav_r = 1
        # uav_f = action.size - uav_r
        fraction = np.zeros([action.size, 1])
        index_rel = np.where(action > 0)[0]

        for uav in np.arange(action.size):
            if uav == index_rel:
                fraction[uav] = (power.get('Power_pt') * power.get('Power_UAV_pr') * (abs(h_gt_uav[uav]))**2 *
                                 (abs(h_uav_gr[uav]))**2)/(1 + (power.get('Power_pt') * (abs(h_gt_uav[uav]))**2) +
                                                           (power.get('Power_UAV_pr') * (abs(h_uav_gr[uav]))**2))
            else:
                fraction[uav] = (power.get('Power_source') * power.get('Power_fusion') * (abs(h_s_uav[uav])) ** 2 *
                                 (abs(h_uav_f[uav])) ** 2)/(1 + (power.get('Power_source') * (abs(h_s_uav[uav])) ** 2) +
                                                            (power.get('Power_fusion')) * (abs(h_uav_f[uav])) ** 2)
        total = 0.5 * np.log2(1 + fraction)

    return np.squeeze(total)
