"""
Created on June 26, 2019
@author: Alireza Shamsoshoara
@Project: Multi Agent Reinforcement Learning (Independent Agents)
          An Autonomous Spectrum Management Scheme for Unmanned Aerial Vehicle Networks in Disaster Relief Operations
          Paper: https://ieeexplore.ieee.org/abstract/document/9046033
          Arxiv: https://arxiv.org/abs/1911.11343
@Northern Arizona University
This project is developed and tested with Python 3.6 using pycharm on Ubuntu 18.04
"""

#########################################################
# import libraries
# General Modules
import os
import time
import numpy as np
from random import seed
from copy import deepcopy
import matplotlib.pyplot as plt

# Customized Modules
from config import Size
from config import Config_Dim as Dim
from config import Config_General as General

from config import Config_RL
from config import Config_Path
from config import Float_Precision
from config import Config_Power as Power

from csi import get_csi
import csi as csi_module
import location_gen as loc
from utility_general import singleutil
from gain_util_jain import reward_val_single
from statefromloc import getstateloc
from action_sel import action_explore
from action_sel import action_exploit
from findq import find_max_q_next_single
from energy import init_energy
from energy import update_energy_movement
from energy import update_energy_transmission
from basestation import regioncenter
from basestation import search_primary_region
from basestation import search_primary_uav
from basestation import search_secondary_uav
from basestation import random_primary_uav
from basestation import random_secondary_uav_search_region
from basestation import random_primary_region
from basestation import random_secondary_uav_random_region
from basestation import search_secondary_uav_random_region
from plotlocation import update2d_figure
from statefromloc import get_region_state_from_general
from statefromloc import get_general_loc_from_region

#########################################################
# General Flags
Flag_Print = False

#########################################################
# Scenario Definition
print(General, "Size = ", Size)
num_UAV = General.get('NUM_UAV')
num_Eps = General.get('NUM_EPS')
num_Step = General.get('NUM_STEP')
num_Pkt = General.get('NUM_PKT')
num_Run = General.get('NUM_RUN')
Region = Dim.get('region')

pathDist = Config_Path.get('PathDist')
pathH = Config_Path.get('PathH')
pathEnergy = Config_Path.get('pathEnergy')

location_init, fig_loc, ax_loc = \
    loc.location(num_UAV,
                 Dim.get('Height'), Dim.get('Length'), Dim.get('Width'),
                 Dim.get('UAV_L_MAX'), Dim.get('UAV_L_MIN'),
                 Dim.get('UAV_W_MAX'), Dim.get('UAV_W_MIN'), pathDist,
                 General.get('Location_SaveFile'), General.get('PlotLocation'), Dim.get('Divider'))

loc_dict = deepcopy(location_init)
Length = Dim.get('Length')
Width = Dim.get('Width')
Height = Dim.get('Height')
Divider = Dim.get('Divider')
region_length = int(Size/int(np.sqrt(Region)))

CSI_Param = csi_module.load_csi(num_UAV, loc_dict, pathH, General.get('CSI_SaveFile'))
energy_init = init_energy(num_UAV, Power.get('MinEnergy'), Power.get('MaxEnergy'), General.get('Energy_SaveFile'),
                          pathEnergy)
energy_init = np.round(energy_init, Float_Precision)
reg_center, reg_ids = regioncenter(Size, Region, Dim.get('Height'))
# Find the nearest region to the Primary receiver (Checked!)
min_reg_dist, min_reg_idx, primary_region_id = search_primary_region(reg_center, loc_dict, reg_ids)

if General.get('Mode') == 2 or General.get('Mode') == 3:
    min_reg_idx, primary_region_id = random_primary_region(reg_ids)

residual_energy_after_flight = np.zeros([num_UAV, 1], dtype=float)
pu_uav_energy_idx = -1

if General.get('PlotLocation'):
    figure_2d_init = update2d_figure(None, location_init, Width, region_length)

# Normal Mode for all UAVs
if General.get('Mode') == 0 or General.get('Mode') == 3 or General.get('Mode') == 4:
    # Do search to find the best UAV for the Primary user (Just one UAV)(Consider residual energy and the distance for
    # all UAVs to the primary user region)
    nearest_uav_idx, pu_uav_energy_idx, residual_energy_after_flight = search_primary_uav(reg_center, min_reg_idx,
                                                                                          primary_region_id, loc_dict,
                                                                                          num_UAV, energy_init,
                                                                                          Power.get('mob_consump_inter')
                                                                                          )

# Do random allocation for the primary uav based on the primary region
if General.get('Mode') == 1 or General.get('Mode') == 2:
    pu_uav_energy_idx, residual_energy_after_flight = random_primary_uav(reg_center, min_reg_idx, loc_dict, num_UAV,
                                                                         energy_init, Power.get('mob_consump_inter'))

# Update the energy for the primary uav which moved from its initial location to the primary region
energy_before_RL = np.zeros([num_UAV], dtype=float)
energy_before_RL[:] = deepcopy(energy_init)  # Initial energy
energy_before_RL[pu_uav_energy_idx] = residual_energy_after_flight[pu_uav_energy_idx]  # Initial Energy after the PU
# Flight

# Update the location for the primary uav which moved from its initial to the primary region
X_U = loc_dict.get('X_U')
Y_U = loc_dict.get('Y_U')
Z_U = loc_dict.get('Z_U')

X_U[pu_uav_energy_idx] = reg_center[min_reg_idx, 0]
Y_U[pu_uav_energy_idx] = reg_center[min_reg_idx, 1]
Z_U[pu_uav_energy_idx] = reg_center[min_reg_idx, 2]

all_uavs_region = np.zeros([num_UAV], dtype=int)
su_uav_id = np.zeros([num_UAV-1], dtype=int)

# Do Search in the UAV database to locate them in the proper regions (rest of the UAVs). It seems that it can be solved
# using biportite maximum graph matching problem
# Initialize Energy after the SUs Flight
# Normal Mode for all UAVs
if General.get('Mode') == 0 or General.get('Mode') == 4:
    energy_before_RL[:], all_uavs_region, su_uav_id, su_regions_id = \
        search_secondary_uav(num_UAV, min_reg_idx, primary_region_id, pu_uav_energy_idx, reg_center,
                             energy_before_RL[:], loc_dict, Power.get('mob_consump_inter'), reg_ids)

if General.get('Mode') == 1:
    energy_before_RL[:], all_uavs_region, su_uav_id, su_regions_id = \
        random_secondary_uav_search_region(num_UAV, min_reg_idx, pu_uav_energy_idx, reg_center, energy_before_RL[:],
                                           loc_dict, Power.get('mob_consump_inter'))

if General.get('Mode') == 2:
    energy_before_RL[:], all_uavs_region, su_uav_id, su_regions_id = \
        random_secondary_uav_random_region(num_UAV, min_reg_idx, pu_uav_energy_idx, reg_center, energy_before_RL[:],
                                           loc_dict, Power.get('mob_consump_inter'), reg_ids)

if General.get('Mode') == 3:
    energy_before_RL[:], all_uavs_region, su_uav_id, su_regions_id = \
        search_secondary_uav_random_region(num_UAV, min_reg_idx, pu_uav_energy_idx, reg_center, energy_before_RL[:],
                                           loc_dict, Power.get('mob_consump_inter'), reg_ids)

# updated_fig2 = update3d_figure(updated_fig, loc_dict, Height, Width)
# figure_2d2 = update2d_figure(None, location_init, Width)
if General.get('PlotLocation'):
    figure_2d = update2d_figure(None, loc_dict, Width, region_length)

gamma = Config_RL.get('gamma')
alpha = Config_RL.get('alpha')
epsilon = Config_RL.get('epsilon')
const_greedy = Config_RL.get('const_greedy')

num_F = 1
num_R = num_UAV - num_F

Dim_L = Length
Dim_W = Width
num_states = Dim_L * Dim_W
num_action = 5  # 0: Up, 1: down, 2: Left, 3: Right, 4: No Movement
action_list = [0, 1, 2, 3, 4]  # 0: Up, 1: down, 2: Left, 3: Right, 4: No Movement

num_Region_states = int(Size / int(np.sqrt(Region)))**2
region_size = int(np.sqrt(num_Region_states))
#########################################################
# Initialization
for Run in range(0, num_Run):
    u_primary = np.zeros([num_Eps, num_Step])
    u_fusion = np.zeros([num_Eps, num_Step])
    u_network = np.zeros([num_Eps, num_Step, num_UAV])
    sum_utility = np.zeros([num_Eps, num_Step])
    reward = np.zeros([num_Eps, num_Step, num_UAV])

    delta_upn = np.zeros([num_Eps, num_Step])
    delta_ufn = np.zeros([num_Eps, num_Step])
    delta_up = np.zeros([num_Eps, num_Step])
    delta_un = np.zeros([num_Eps, num_Step])
    delta_utility = np.zeros([num_Eps, num_Step, num_UAV])

    jainVal = np.zeros([num_Eps, num_Step])
    jain_scaled = np.zeros([num_Eps, num_Step])

    state_new_reg = np.zeros([num_UAV], dtype=int)

    # TODO: Checked!: Zero Values for energy array.
    energy = np.zeros([num_Eps, num_Step, num_UAV], dtype=float)
    #########################################################
    # Initialization for the MA RL algorithm

    X_Mat = np.zeros([num_Eps, num_Step, num_UAV], dtype=int)
    Y_Mat = np.zeros([num_Eps, num_Step, num_UAV], dtype=int)

    X_Mat_Reg = np.zeros([num_Eps, num_Step, num_UAV], dtype=int)
    Y_Mat_Reg = np.zeros([num_Eps, num_Step, num_UAV], dtype=int)

    State_Mat = np.zeros([num_Eps, num_Step, num_UAV], dtype=int)
    next_state_index = np.zeros([num_Eps, num_Step, num_UAV], dtype=int)
    next_state_index_region = np.zeros([num_Eps, num_Step, num_UAV], dtype=int)

    action = np.zeros([num_Eps, num_Step, num_UAV], dtype=int)
    task_matrix = np.zeros([num_Eps, num_Step, num_UAV], dtype=int)
    prev_task = np.zeros([num_Eps, num_Step, num_UAV], dtype=int)
    task_diff = np.zeros([num_Eps, num_Step], dtype=int)

    seed(a=None)
    #########################################################
    # Main Function of the Simulation
    qVal = np.zeros([num_UAV, num_Region_states, num_action])
    timer = 0
    for Eps in range(0, num_Eps):
        timer = time.clock()
        number_meet = np.zeros([num_UAV, num_Region_states], dtype=int)
        X_U = deepcopy(loc_dict.get('X_U'))
        Y_U = deepcopy(loc_dict.get('Y_U'))
        # TODO: Checked!: Reading the initial values for the energy like the location.
        energy[Eps, 0, :] = deepcopy(energy_before_RL)

        for Step in range(0, num_Step):
            if num_Eps == 1:
                print(" -----------------Epoch = %d,  Step = %d ----------------- " % (Eps, Step))
            X_Mat[Eps, Step, :] = np.squeeze(X_U)
            Y_Mat[Eps, Step, :] = np.squeeze(Y_U)
            state_index_general = getstateloc(X_U, Y_U, Size)
            state_index_region, X_Mat_Reg[Eps, Step, :], Y_Mat_Reg[Eps, Step, :] = \
                get_region_state_from_general(state_index_general, Size, region_size)

            X_U_Reg = deepcopy(X_Mat_Reg[Eps, Step, :])
            Y_U_Reg = deepcopy(Y_Mat_Reg[Eps, Step, :])

            exploration_current_state = deepcopy(state_index_region)
            State_Mat[Eps, Step, :] = np.squeeze(exploration_current_state)
            number_meet[:, state_index_region] += 1

            if Step > 0:
                prev_task[Eps, Step, :] = task_matrix[Eps, Step-1, :].copy(order='C')

            for UAV in range(0, num_UAV):

                if np.random.rand() < epsilon or General.get('Mode') == 4:
                    ###################
                    # Exploration
                    action[Eps, Step, UAV], X_U_Reg[UAV], Y_U_Reg[UAV], state_new_reg[UAV] = \
                        action_explore(X_U[UAV], Y_U[UAV], action_list, Size, exploration_current_state[UAV], UAV,
                                       region_size, X_U_Reg[UAV], Y_U_Reg[UAV])

                    X_U[UAV], Y_U[UAV] = get_general_loc_from_region(X_U_Reg[UAV], Y_U_Reg[UAV], all_uavs_region[UAV],
                                                                     region_size, Size)

                else:
                    ###################
                    # Exploitation
                    action[Eps, Step, UAV], X_U_Reg[UAV], Y_U_Reg[UAV], state_new_reg[UAV] = \
                        action_exploit(X_U[UAV], Y_U[UAV], action_list, Size, exploration_current_state[UAV],
                                       qVal[UAV, :, :], region_size, X_U_Reg[UAV], Y_U_Reg[UAV])

                    X_U[UAV], Y_U[UAV] = get_general_loc_from_region(X_U_Reg[UAV], Y_U_Reg[UAV], all_uavs_region[UAV],
                                                                     region_size, Size)

                if UAV in su_uav_id:  # Fusion = 0
                    task_matrix[Eps, Step, UAV] = 0
                else:  # Relay = 1
                    task_matrix[Eps, Step, UAV] = 1

            task_diff[Eps, Step] = np.sum(np.not_equal(task_matrix[Eps, Step, :], prev_task[Eps, Step, :]))

            # TODO: Checked!: Updating the energy for each UAV after changing its location
            if Step > 0:
                energy[Eps, Step, :] = energy[Eps, Step-1, :]
            energy[Eps, Step, :] = update_energy_movement(energy[Eps, Step, :], action[Eps, Step, :],
                                                          Power.get('mob_consump_intra'))

            if Flag_Print:
                print(' Current General State = ', np.squeeze(state_index_general), '\n Current Region State = ',
                      np.squeeze(state_index_region))
                print(' Current X = ', X_Mat[Eps, Step, :])
                print(' Current Y = ', Y_Mat[Eps, Step, :])
                print(' Actions = ', np.squeeze(action[Eps, Step, :]))
                print(' New Region State = ', state_new_reg)
                print(' New X = ', np.squeeze(X_U))
                print(' New Y = ', np.squeeze(Y_U))
                print(' New X_Reg = ', X_U_Reg)
                print(' New Y_Reg = ', Y_U_Reg)
                print(' Tasks = ', np.squeeze(task_matrix[Eps, Step, :]))
            #################################
            # Updating utilities
            csi_coef = get_csi(num_UAV, loc_dict, np.squeeze(X_U), np.squeeze(Y_U))
            u_network[Eps, Step, :] = singleutil(task_matrix[Eps, Step, :], csi_coef, General, Power)

            # TODO: Checked!: Updating the energy for each UAV after transmission
            energy[Eps, Step, :] = update_energy_transmission(energy[Eps, Step, :], Power.get('trans_consump'))

            uav_r = np.sum(task_matrix[Eps, Step, :])  # 1 = Relay, 0 = Fusion
            uav_r = int(uav_r)
            uav_f = int(General.get('NUM_UAV') - uav_r)

            sum_utility[Eps, Step] = np.sum(u_network[Eps, Step, :])

            # TODO: Checked!: Consider the energy in Rewarding to have less mobility, in order to save more energy
            if Step == 0:
                reward[Eps, Step, :] = \
                    reward_val_single(u_network[Eps, Step, :], u_network[Eps, 0, :], sum_utility[Eps, Step],
                                      sum_utility[Eps, 0], energy[Eps, Step, :], energy_before_RL,
                                      Power.get('mob_consump_intra'))
            else:
                reward[Eps, Step, :] = \
                    reward_val_single(u_network[Eps, Step, :], u_network[Eps, 0: Step, :], sum_utility[Eps, Step],
                                      sum_utility[Eps, 0: Step], energy[Eps, Step, :], energy[Eps, Step-1, :],
                                      Power.get('mob_consump_intra'))

            if Flag_Print:
                print(" Utility Values = ", np.squeeze(u_network[Eps, Step, :]))
                print(" SUM Utility = ", sum_utility[Eps, Step])
                print(" Energy = ", np.squeeze(energy[Eps, Step, :]))
                print("Reward = ", reward[Eps, Step, :])

            #################################
            # Updating Q-Table and Q values
            next_state_index[Eps, Step, :] = np.squeeze(getstateloc(X_U, Y_U, Size))
            next_state_index_region[Eps, Step, :], _, _ = get_region_state_from_general(next_state_index[Eps, Step, :],
                                                                                        Size, region_size)
            maxQ_NextState = np.zeros([num_UAV, 1], dtype=float)
            for uav in np.arange(num_UAV):
                maxQ_NextState[uav] = find_max_q_next_single(qVal[uav, :, :], next_state_index_region[Eps, Step, uav],
                                                             X_U_Reg[uav], Y_U_Reg[uav], action_list, region_size)
                qVal[uav, State_Mat[Eps, Step, uav], action[Eps, Step, uav]] = \
                    (1-alpha) * qVal[uav, State_Mat[Eps, Step, uav], action[Eps, Step, uav]] + alpha * \
                    (reward[Eps, Step, uav] + gamma * maxQ_NextState[uav])
                if Flag_Print:
                    print("QVal = ", np.squeeze(qVal[uav, State_Mat[Eps, Step, uav], action[Eps, Step, uav]]))

            # End of the Each Step

        print(" -------Run = %d ----- Epoch = %d ----------------- Duration = %f " % (Run, Eps, time.clock() - timer))
        # ********************************
        # End of the Each Episode

    if General.get('PlotResult'):
        if num_Eps > 1:
            plt.figure()
            plt.plot(range(0, num_Eps), np.sum(sum_utility, axis=1), markersize='10', color='blue')
            plt.grid(True)
            plt.ylabel('Sum Utility')
            plt.xlabel('Episodes')
            plt.show(block=False)

            plt.figure()
            plt.plot(range(0, num_Eps), 100 * np.sum(u_network[:, :, 4], axis=1), markersize='5')
            plt.grid(True)
            plt.ylabel('individual Utility')
            plt.xlabel('Episodes')
            plt.show(block=False)

        else:
            plt.figure()
            plt.plot(range(0, num_Step), np.mean(reward[:, :, 0], axis=0), markersize='10', color='blue')
            plt.grid(True)
            plt.ylabel('reward')
            plt.xlabel('Steps')
            plt.savefig('first.png')
            plt.show(block=False)

            plt.figure()
            plt.plot(range(0, num_Step), np.mean(task_diff, axis=1), markersize='10', color='blue')
            plt.grid(True)
            plt.ylabel('Number of Switch')
            plt.xlabel('Steps')
            # plt.savefig('first.png')
            plt.show(block=False)

            plt.figure()
            plt.plot(range(0, num_Step), sum_utility[199, :], markersize='10')
            plt.grid(True)
            plt.ylabel('Sum Utility')
            plt.xlabel('Steps')
            # plt.savefig('first.png')
            plt.show(block=False)

            plt.figure()
            plt.plot(range(0, num_Step), np.mean(np.cumsum(reward, axis=0), axis=1), markersize='10', color='blue')
            plt.grid(True)
            plt.ylabel('Accumulative reward')
            plt.xlabel('Steps')
            # plt.savefig('first.png')
            plt.show(block=False)

    if General.get('SaveOutput'):
        path_save = 'SimulationData/sensor2019/Mode_%d/Grid_Size_%d' % (General.get('Mode'), Size)
        if not os.path.exists(path_save):
            os.makedirs(path_save)

        # outputFile_win = 'C:\SimulationData\Out_greedy_Size_%d_Run_%d_Eps_%d_Step_%d.npz' % (Size, Run, num_Eps,
        # num_Step)
        outputFile_linux = \
            'SimulationData/sensor2019/Mode_%d/Grid_Size_%d/Out_UAV_%d_greedy_Size_%d_Region_%d_Run_%d_' \
            'Eps_%d_Step_%d.npz' % (General.get('Mode'), Size, num_UAV, Size, Region, Run, num_Eps, num_Step)

        np.savez(outputFile_linux, u_network=u_network, sum_utility=sum_utility, reward=reward, X_Mat=X_Mat,
                 Y_Mat=Y_Mat, X_Mat_Reg=X_Mat_Reg, Y_Mat_Reg=Y_Mat_Reg, energy=energy, State_Mat=State_Mat,
                 action=action, task_matrix=task_matrix, next_state_index=next_state_index, qVal=qVal,
                 next_state_index_region=next_state_index_region)
    # End of the Each Run

seed(1)
