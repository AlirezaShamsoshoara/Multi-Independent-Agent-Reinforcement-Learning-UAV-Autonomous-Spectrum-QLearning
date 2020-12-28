#################################
# Load the npz file from the local drive and plot the results
#################################

#########################################################
# import libraries
import numpy as np
import matplotlib.pyplot as plt
import time
from copy import deepcopy
from config import Config_Power
#########################################################
# Function definition
Size_list = [9, 16, 27, 32, 36, 64, 81]
Region_list = [9, 16, 9, 64, 16, 16, 9]
Step_list = [75, 125, 600, 125, 600, 2000, 6000]
Mode_list = [0, 1, 2, 3, 4]
num_Run = 20
Run_list = range(0, num_Run)
num_Eps = 40
num_UAV = 5
uav_list = range(0, num_UAV)
Size = Size_list.index(81)
num_step = Step_list[Size]
part = 2
num_Mode = Mode_list.__len__()
transmission_rate = Config_Power.get('trans_consump')

linestyle_tuple = [
     ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 1))),
     ('densely dotted',        (0, (1, 1))),

     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]


def normalize(d):
    # d is a (n x dimension) np array
    d -= np.min(d, axis=0)
    d /= np.ptp(d, axis=0)
    return d


# ********************************************************************* First part of the simulation
def first_part():
    # ********************************************************************* GRID SIZE = 9 =  9 x 9
    if Size_list[Size] == 9:
        sum_utility_step = np.zeros([num_Run, num_Eps], dtype=float)
        u_network_step = np.zeros([num_Run, num_Eps, num_UAV], dtype=float)
        action_array = np.zeros([num_Run, num_Eps, num_step, num_UAV], dtype=int)
        movement = np.zeros([num_Run, num_Eps], dtype=int)
        movement_uav = np.zeros([num_Run, num_Eps, num_UAV], dtype=int)
        energy = np.zeros([num_Run, num_Eps, num_step, num_UAV], dtype=float)
        energy_consumption_rate_uav = np.zeros([num_Run, num_Eps, num_UAV], dtype=float)
        reward = np.zeros([num_Run, num_Eps, num_step, num_UAV], dtype=float)

        for Run in Run_list:
            outputfile =\
                'SimulationData/sensor2019/Mode_0/Grid_Size_%d/Out_UAV_%d_greedy_Size_%d_Region_%d_Run_%d_Eps_%d' \
                '_Step_%d.npz'\
                % (Size_list[Size], num_UAV, Size_list[Size], Region_list[Size], Run, num_Eps, Step_list[Size])
            readfile = np.load(outputfile)
            sum_utility_step[Run, :] = np.sum(readfile['sum_utility'], axis=1)
            action_array[Run, :, :, :] = readfile['action']
            energy[Run, :, :, :] = readfile['energy']
            reward[Run, :, :, :] = readfile['reward']

            for uav in uav_list:
                u_network_step[Run, :, uav] = np.sum(readfile['u_network'][:, :, uav], axis=1)

            for eps in range(0, num_Eps):
                for step in range(0, Step_list[Size]):
                    for uav in uav_list:
                        if action_array[Run, eps, step, uav] < 4:
                            movement[Run, eps] += 1
                            movement_uav[Run, eps, uav] += 1

            for Eps in range(0, num_Eps):
                for uav in uav_list:
                    energy_consumption_rate_uav[Run, Eps, uav] = energy[Run, Eps, 0, uav] - \
                                                                 energy[Run, Eps, int(num_step * 0.75), uav]

        energy_mean = np.mean(energy, axis=0)
        min_energ_mean = np.min(energy_mean, axis=1)
        argmin_energy_mean = np.argmin(energy_mean, axis=1)
        lifetime = deepcopy(argmin_energy_mean)
        lifetime = lifetime + (min_energ_mean/transmission_rate).astype(int) - 1
        task_matrix = readfile['task_matrix'][0, 0, :]
        print('Task Matrix = ', task_matrix)
        reward_mean = np.mean(reward, axis=0)
        reward_mean_sum = np.sum(reward_mean, axis=1)

        #  ******************* Plotting Sum Utility
        plt.figure()
        plt.plot(range(0, num_Eps), np.mean(sum_utility_step, axis=0), markersize='10', linewidth=2.0, color='blue',
                 label="Sum Utility")
        plt.grid(True)
        plt.ylabel('Sum Utility', fontsize=14, fontweight="bold")
        plt.xlabel('Episodes', fontsize=14, fontweight="bold")
        plt.title('Sum utility %d x %d' % (Size_list[Size], Size_list[Size]))
        plt.legend(prop={'size': 14})
        plt.show(block=False)
        #  ****************************************
        #  ******************* Plotting Individual Utility All-in-One
        plt.figure()
        for uav in uav_list:
            plt.plot(range(0, num_Eps), np.mean(u_network_step[:, :, uav], axis=0), markersize='10', linewidth=2.0,
                     label="UAV[%d] Utility" % uav)
        plt.grid(True)
        plt.ylabel('UAV Utility', fontsize=14, fontweight="bold")
        plt.xlabel('Episodes', fontsize=14, fontweight="bold")
        plt.title('UAV utility %d x %d' % (Size_list[Size], Size_list[Size]))
        plt.legend(prop={'size': 14})
        plt.show(block=False)
        #  ****************************************
        #  ******************* Plotting Individual Utility Different windows
        for uav in uav_list:
            plt.figure()
            plt.plot(range(0, num_Eps), np.mean(u_network_step[:, :, uav], axis=0), markersize='10', linewidth=2.0,
                     label="UAV[%d] Utility" % uav)
            plt.grid(True)
            plt.ylabel('UAV Utility', fontsize=14, fontweight="bold")
            plt.xlabel('Episodes', fontsize=14, fontweight="bold")
            plt.title('UAV utility %d x %d' % (Size_list[Size], Size_list[Size]))
            plt.legend(prop={'size': 14})
            plt.show(block=False)
        #  ****************************************
        #  ******************* Plotting Individual Utility same window different axes
        plt.figure()
        d0 = plt.plot(range(0, num_Eps), np.mean(u_network_step[:, :, 0], axis=0), markersize='4', linewidth=2.0,
                      label="UAV[%d]" % 0, linestyle='-', color='red')
        d1 = plt.plot(range(0, num_Eps), np.mean(u_network_step[:, :, 1], axis=0), markersize='4', linewidth=2.0,
                      label="UAV[%d]" % 1, linestyle='--', color='green')
        d2 = plt.plot(range(0, num_Eps), np.mean(u_network_step[:, :, 2], axis=0), markersize='4', linewidth=2.0,
                      label="UAV[%d]" % 2, linestyle='-.', color='blue')
        d3 = plt.plot(range(0, num_Eps), np.mean(u_network_step[:, :, 3], axis=0), markersize='4', linewidth=2.0,
                      label="UAV[%d]" % 3, linestyle=':', color='magenta', marker='x')
        plt.grid(True)
        plt.ylabel('UAV Utility[0-3]', fontsize=14, fontweight="normal")
        plt.xlabel('Episodes', fontsize=14, fontweight="normal")
        plt.title('UAV utility %d x %d' % (Size_list[Size], Size_list[Size]))

        plt2 = plt.twinx()
        d4 = plt2.plot(range(0, num_Eps), np.mean(u_network_step[:, :, 4], axis=0), markersize='4', linewidth=2.0,
                       label="UAV[%d]" % 4, linestyle='--', color='black', dashes=(5, 2, 10, 2), marker='o')
        plt.ylabel('UAV[4] Utility (--)', fontsize=14, color='black')
        plt_lines = d0 + d1 + d2 + d3 + d4
        label_text = [line.get_label() for line in plt_lines]
        plt.legend(plt_lines, label_text, loc=5, prop={'size': 14})
        plt.show(block=False)
        plt.savefig('Figures/Utility_all_size_%d.pdf' % Size_list[Size], bbox_inches='tight')
        #  ****************************************
        #  ******************* Plotting Individual Utility same window Normalized Axes
        plt.figure()
        d0 = plt.plot(range(0, num_Eps), normalize(np.mean(u_network_step[:, :, 0], axis=0)), markersize='4',
                      label="UAV[%d]" % 0, linestyle='-', color='red', linewidth=2.0)
        d1 = plt.plot(range(0, num_Eps), normalize(np.mean(u_network_step[:, :, 1], axis=0)), markersize='4',
                      label="UAV[%d]" % 1, linestyle='--', color='green', linewidth=2.0)
        d2 = plt.plot(range(0, num_Eps), normalize(np.mean(u_network_step[:, :, 2], axis=0)), markersize='4',
                      label="UAV[%d]" % 2, linestyle='-.', color='blue', linewidth=2.0)
        d3 = plt.plot(range(0, num_Eps), normalize(np.mean(u_network_step[:, :, 3], axis=0)), markersize='4',
                      label="UAV[%d]" % 3, linestyle=':', color='magenta', marker='x', linewidth=2.0)
        d4 = plt.plot(range(0, num_Eps), normalize(np.mean(u_network_step[:, :, 4], axis=0)), markersize='4',
                      label="UAV[%d]" % 4, linestyle=':', color='black', marker='o', linewidth=2.0)
        plt.grid(True)
        plt.ylabel('Normalized UAV Utility', fontsize=14, fontweight="normal")
        plt.xlabel('Episodes', fontsize=14, fontweight="normal")
        plt.title('UAV utility %d x %d' % (Size_list[Size], Size_list[Size]))

        plt_lines = d0 + d1 + d2 + d3 + d4
        label_text = [line.get_label() for line in plt_lines]
        plt.legend(plt_lines, label_text, loc=4, prop={'size': 14})
        plt.show(block=False)
        plt.savefig('Figures/Utility_all_size_%d.pdf' % Size_list[Size], bbox_inches='tight')
        #  ****************************************
        #  ******************* Plotting movement_uav same window
        plt.figure()
        d0 = plt.plot(range(0, num_Eps), np.mean(movement_uav[:, :, 0], axis=0), markersize='4',
                      label="UAV[%d]" % 0, linestyle='-', color='red', linewidth=2.0)
        d1 = plt.plot(range(0, num_Eps), np.mean(movement_uav[:, :, 1], axis=0), markersize='4',
                      label="UAV[%d]" % 1, linestyle='--', color='green', linewidth=2.0)
        d2 = plt.plot(range(0, num_Eps), np.mean(movement_uav[:, :, 2], axis=0), markersize='4',
                      label="UAV[%d]" % 2, linestyle='-.', color='blue', linewidth=2.0)
        d3 = plt.plot(range(0, num_Eps), np.mean(movement_uav[:, :, 3], axis=0), markersize='4',
                      label="UAV[%d]" % 3, linestyle=':', color='magenta', marker='x', linewidth=2.0)
        d4 = plt.plot(range(0, num_Eps), np.mean(movement_uav[:, :, 4], axis=0), markersize='4',
                      label="UAV[%d]" % 4, linestyle=':', color='black', marker='o', linewidth=2.0)
        plt.grid(True)
        plt.ylabel('UAVs mobility', fontsize=14, fontweight="normal")
        plt.xlabel('Episodes', fontsize=14, fontweight="normal")
        plt.title('UAV Mobility %d x %d' % (Size_list[Size], Size_list[Size]))

        plt_lines = d0 + d1 + d2 + d3 + d4
        label_text = [line.get_label() for line in plt_lines]
        plt.legend(plt_lines, label_text, loc=5, prop={'size': 14})
        plt.show(block=False)
        plt.savefig('Figures/mobility_all_size_%d.pdf' % Size_list[Size], bbox_inches='tight')
        #  ****************************************
        # ******************* Plotting Accumulative reward uav same window
        plt.figure()
        d0 = plt.plot(range(0, num_Eps), reward_mean_sum[:, 0], markersize='4',
                      label="UAV[%d]" % 0, linestyle='-', color='red', linewidth=2.0)
        d1 = plt.plot(range(0, num_Eps), reward_mean_sum[:, 1], markersize='4',
                      label="UAV[%d]" % 1, linestyle='--', color='green', linewidth=2.0)
        d2 = plt.plot(range(0, num_Eps), reward_mean_sum[:, 2], markersize='4',
                      label="UAV[%d]" % 2, linestyle='-.', color='blue', linewidth=2.0)
        d3 = plt.plot(range(0, num_Eps), reward_mean_sum[:, 3], markersize='4',
                      label="UAV[%d]" % 3, linestyle=':', color='magenta', marker='x', linewidth=2.0)
        d4 = plt.plot(range(0, num_Eps), reward_mean_sum[:, 4], markersize='4',
                      label="UAV[%d]" % 4, linestyle=':', color='black', marker='o', linewidth=2.0)
        plt.grid(True)
        plt.ylabel('Accumulative Reward', fontsize=14, fontweight="normal")
        plt.xlabel('Episodes', fontsize=14, fontweight="normal")
        plt.title('UAV Accumulative Reward in %d x %d' % (Size_list[Size], Size_list[Size]))

        plt_lines = d0 + d1 + d2 + d3 + d4
        label_text = [line.get_label() for line in plt_lines]
        plt.legend(plt_lines, label_text, loc=4, prop={'size': 14})
        plt.show(block=False)
        plt.savefig('Figures/reward_all_size_%d.pdf' % Size_list[Size], bbox_inches='tight')
        #  ****************************************
        # ******************* Plotting Energy Consumption Rate per uav same window
        plt.figure()
        d0 = plt.plot(range(0, num_Eps), np.mean(energy_consumption_rate_uav[:, :, 0], axis=0), markersize='4',
                      label="UAV[%d]" % 0, linestyle='-', color='red', linewidth=2.0)
        d1 = plt.plot(range(0, num_Eps), np.mean(energy_consumption_rate_uav[:, :, 1], axis=0), markersize='4',
                      label="UAV[%d]" % 1, linestyle='--', color='green', linewidth=2.0)
        d2 = plt.plot(range(0, num_Eps), np.mean(energy_consumption_rate_uav[:, :, 2], axis=0), markersize='4',
                      label="UAV[%d]" % 2, linestyle='-.', color='blue', linewidth=2.0)
        d3 = plt.plot(range(0, num_Eps), np.mean(energy_consumption_rate_uav[:, :, 3], axis=0), markersize='4',
                      label="UAV[%d]" % 3, linestyle=':', color='magenta', marker='x', linewidth=2.0)
        d4 = plt.plot(range(0, num_Eps), np.mean(energy_consumption_rate_uav[:, :, 4], axis=0), markersize='4',
                      label="UAV[%d]" % 4, linestyle=':', color='black', marker='o', linewidth=2.0)
        plt.grid(True)
        plt.ylabel('Consumption Rate', fontsize=14, fontweight="normal")
        plt.xlabel('Episodes', fontsize=14, fontweight="normal")
        plt.title('Energy Consumption Rate in %d x %d' % (Size_list[Size], Size_list[Size]))

        plt_lines = d0 + d1 + d2 + d3 + d4
        label_text = [line.get_label() for line in plt_lines]
        plt.legend(plt_lines, label_text, loc=1, prop={'size': 14})
        plt.show(block=False)
        plt.savefig('Figures/consumption_all_size_%d.pdf' % Size_list[Size], bbox_inches='tight')
        #  ****************************************
        plt.figure()
        plt.plot(range(0, num_Eps), np.mean(movement, axis=0), markersize='10', color='red')
        plt.grid(True)
        plt.ylabel('Movement actions')
        plt.xlabel('Episodes')
        plt.title('Number of movements in each episode for the whole network(All UAVs)')
        plt.show(block=False)
        #  ****************************************
        plt.figure()
        for uav in uav_list:
            plt.plot(range(0, num_Eps), np.mean(energy_consumption_rate_uav[:, :, uav], axis=0), markersize='10',
                     linewidth=2.0, label="UAV[%d]" % uav)
        plt.grid(True)
        plt.ylabel('Energy consumption rate per episode (j)')
        plt.xlabel('Episodes')
        plt.title('Energy consumption rate for each UAV')
        plt.show(block=False)
        plt.legend(prop={'size': 14})
        #  ****************************************
        # ******************* Plotting Lifetime per uav same window
        plt.figure()
        d0 = plt.plot(range(0, num_Eps), lifetime[:, 0], markersize='4',
                      label="UAV[%d]" % 0, linestyle='-', color='red', linewidth=2.0)
        d1 = plt.plot(range(0, num_Eps), lifetime[:, 1], markersize='4',
                      label="UAV[%d]" % 1, linestyle='--', color='green', linewidth=2.0)
        d2 = plt.plot(range(0, num_Eps), lifetime[:, 2], markersize='4',
                      label="UAV[%d]" % 2, linestyle='-.', color='blue', linewidth=2.0)
        d3 = plt.plot(range(0, num_Eps), lifetime[:, 3], markersize='4',
                      label="UAV[%d]" % 3, linestyle=':', color='magenta', marker='x', linewidth=2.0)
        d4 = plt.plot(range(0, num_Eps), lifetime[:, 4], markersize='4',
                      label="UAV[%d]" % 4, linestyle=':', color='black', marker='o', linewidth=2.0)
        plt.grid(True)
        plt.ylabel('Lifetime', fontsize=14, fontweight="normal")
        plt.xlabel('Episodes', fontsize=14, fontweight="normal")
        plt.title('Lifetime(Number of transmissions) in %d x %d' % (Size_list[Size], Size_list[Size]))

        plt_lines = d0 + d1 + d2 + d3 + d4
        label_text = [line.get_label() for line in plt_lines]
        plt.legend(plt_lines, label_text, loc=7, prop={'size': 11})
        plt.show(block=False)
        plt.savefig('Figures/lifetime_all_size_%d.pdf' % Size_list[Size], bbox_inches='tight')
        #  ****************************************
        plt.figure()
        for uav in uav_list:
            plt.plot(range(0, num_Eps), lifetime[:, uav], markersize='10', linewidth=2.0, label="UAV[%d]" % uav)
        plt.grid(True)
        plt.ylabel('Lifetime')
        plt.xlabel('Episodes')
        plt.title('Successful transmission time before UAV battery depletion')
        plt.show(block=False)
        plt.legend(prop={'size': 14})
        #  ****************************************
        for uav in uav_list:
            plt.figure()
            plt.plot(range(0, num_Eps), lifetime[:, uav], markersize='10', linewidth=2.0, label="UAV[%d]" % uav)
            plt.grid(True)
            plt.ylabel('Lifetime')
            plt.xlabel('Episodes')
            plt.title('Successful transmission time before UAV battery depletion')
            plt.show(block=False)
            plt.legend(prop={'size': 14})

        del sum_utility_step, u_network_step, action_array, movement, energy, energy_consumption_rate_uav, lifetime, \
            task_matrix, movement_uav, reward,

    # ********************************************************************* GRID SIZE = 81 =  81 x 81
    if Size_list[Size] == 81:
        sum_utility_step = np.zeros([num_Run, num_Eps], dtype=float)
        u_network_step = np.zeros([num_Run, num_Eps, num_UAV], dtype=float)
        action_array = np.zeros([num_Run, num_Eps, num_step, num_UAV], dtype=int)
        movement = np.zeros([num_Run, num_Eps], dtype=int)
        movement_uav = np.zeros([num_Run, num_Eps, num_UAV], dtype=int)
        energy = np.zeros([num_Run, num_Eps, num_step, num_UAV], dtype=float)
        energy_consumption_rate_uav = np.zeros([num_Run, num_Eps, num_UAV], dtype=float)
        reward = np.zeros([num_Run, num_Eps, num_step, num_UAV], dtype=float)

        for Run in Run_list:
            outputfile =\
                'SimulationData/sensor2019/Mode_0/Grid_Size_%d/Out_UAV_%d_greedy_Size_%d_Region_%d_Run_%d_Eps_%d' \
                '_Step_%d.npz'\
                % (Size_list[Size], num_UAV, Size_list[Size], Region_list[Size], Run, num_Eps, Step_list[Size])
            readfile = np.load(outputfile)
            sum_utility_step[Run, :] = np.sum(readfile['sum_utility'], axis=1)
            action_array[Run, :, :, :] = readfile['action']
            energy[Run, :, :, :] = readfile['energy']
            reward[Run, :, :, :] = readfile['reward']

            for uav in uav_list:
                u_network_step[Run, :, uav] = np.sum(readfile['u_network'][:, :, uav], axis=1)

            for eps in range(0, num_Eps):
                timer = time.clock()
                for step in range(0, Step_list[Size]):
                    for uav in uav_list:
                        if action_array[Run, eps, step, uav] < 4:
                            movement[Run, eps] += 1
                            movement_uav[Run, eps, uav] += 1
                print (" -------Run = %d ----- Epoch = %d ----------------- Duration = %f " % (
                    Run, eps, time.clock() - timer))

            for Eps in range(0, num_Eps):
                for uav in uav_list:
                    energy_consumption_rate_uav[Run, Eps, uav] = energy[Run, Eps, 0, uav] - energy[Run, Eps, -1, uav]

        energy_mean = np.mean(energy, axis=0)
        min_energ_mean = np.min(energy_mean, axis=1)
        argmin_energy_mean = np.argmin(energy_mean, axis=1)
        lifetime = deepcopy(argmin_energy_mean)
        lifetime = lifetime + (min_energ_mean / transmission_rate).astype(int) - 1
        task_matrix = readfile['task_matrix'][0, 0, :]
        print('Task Matrix = ', task_matrix)
        reward_mean = np.mean(reward, axis=0)
        reward_mean_sum = np.sum(reward_mean, axis=1)

        #  ******************* Plotting Sum Utility
        plt.figure()
        plt.plot(range(0, num_Eps), np.mean(sum_utility_step, axis=0), markersize='10', linewidth=2.0, color='blue',
                 label="Sum Utility")
        plt.grid(True)
        plt.ylabel('Sum Utility', fontsize=14, fontweight="normal")
        plt.xlabel('Episodes', fontsize=14, fontweight="normal")
        plt.title('Sum utility %d x %d' % (Size_list[Size], Size_list[Size]))
        plt.legend(prop={'size': 14})
        plt.show(block=False)
        #  ****************************************
        #  ******************* Plotting Individual Utility All-in-One
        plt.figure()
        for uav in uav_list:
            plt.plot(range(0, num_Eps), np.mean(u_network_step[:, :, uav], axis=0), markersize='10', linewidth=2.0,
                     label="UAV[%d] Utility" % uav)
        plt.grid(True)
        plt.ylabel('UAV Utility', fontsize=14, fontweight="normal")
        plt.xlabel('Episodes', fontsize=14, fontweight="normal")
        plt.title('UAV utility %d x %d' % (Size_list[Size], Size_list[Size]))
        plt.legend(prop={'size': 14})
        plt.show(block=False)
        #  ****************************************
        #  ******************* Plotting Individual Utility Different windows
        for uav in uav_list:
            plt.figure()
            plt.plot(range(0, num_Eps), np.mean(u_network_step[:, :, uav], axis=0), markersize='10', linewidth=2.0,
                     label="UAV[%d] Utility" % uav)
            plt.grid(True)
            plt.ylabel('UAV Utility', fontsize=14, fontweight="normal")
            plt.xlabel('Episodes', fontsize=14, fontweight="normal")
            plt.title('UAV utility %d x %d' % (Size_list[Size], Size_list[Size]))
            plt.legend(prop={'size': 14})
            plt.show(block=False)
        #  ****************************************
        #  ******************* Plotting Individual Utility same window different axes
        plt.figure()
        d0 = plt.plot(range(0, num_Eps), np.mean(u_network_step[:, :, 0], axis=0), markersize='4', linewidth=2.0,
                      label="UAV[%d] Utility" % 0, linestyle='-', color='red')
        d1 = plt.plot(range(0, num_Eps), np.mean(u_network_step[:, :, 1], axis=0), markersize='4', linewidth=2.0,
                      label="UAV[%d] Utility" % 1, linestyle='--', color='green')
        d2 = plt.plot(range(0, num_Eps), np.mean(u_network_step[:, :, 2], axis=0), markersize='4', linewidth=2.0,
                      label="UAV[%d] Utility" % 2, linestyle='-.', color='blue')
        d3 = plt.plot(range(0, num_Eps), np.mean(u_network_step[:, :, 3], axis=0), markersize='4', linewidth=2.0,
                      label="UAV[%d] Utility" % 3, linestyle=':', color='magenta', marker='x')
        plt.grid(True)
        plt.ylabel('UAV Utility[0-3]', fontsize=14, fontweight="normal")
        plt.xlabel('Episodes', fontsize=14, fontweight="normal")
        plt.title('UAV utility %d x %d' % (Size_list[Size], Size_list[Size]))

        plt2 = plt.twinx()
        d4 = plt2.plot(range(0, num_Eps), np.mean(u_network_step[:, :, 4], axis=0), markersize='4', linewidth=2.0,
                       label="UAV[%d] Utility" % 4, linestyle='--', color='black', dashes=(5, 2, 10, 2), marker='o')
        plt.ylabel('UAV[4] Utility (--)', fontsize=14, color='black')
        plt_lines = d0 + d1 + d2 + d3 + d4
        label_text = [line.get_label() for line in plt_lines]
        plt.legend(plt_lines, label_text, loc=5, prop={'size': 14})
        plt.show(block=False)
        plt.savefig('Figures/Utility_all_size_%d.pdf' % Size_list[Size], bbox_inches='tight')
        #  ****************************************
        #  ******************* Plotting Individual Utility same window Normalized Axes
        plt.figure()
        d0 = plt.plot(range(0, num_Eps), normalize(np.mean(u_network_step[:, :, 0], axis=0)), markersize='4',
                      label="UAV[%d] Utility" % 0, linestyle='-', color='red', linewidth=2.0)
        d1 = plt.plot(range(0, num_Eps), normalize(np.mean(u_network_step[:, :, 1], axis=0)), markersize='4',
                      label="UAV[%d] Utility" % 1, linestyle='--', color='green', linewidth=2.0)
        d2 = plt.plot(range(0, num_Eps), normalize(np.mean(u_network_step[:, :, 2], axis=0)), markersize='4',
                      label="UAV[%d] Utility" % 2, linestyle='-.', color='blue', linewidth=2.0)
        d3 = plt.plot(range(0, num_Eps), normalize(np.mean(u_network_step[:, :, 3], axis=0)), markersize='4',
                      label="UAV[%d] Utility" % 3, linestyle=':', color='magenta', marker='x', linewidth=2.0)
        d4 = plt.plot(range(0, num_Eps), normalize(np.mean(u_network_step[:, :, 4], axis=0)), markersize='4',
                      label="UAV[%d] Utility" % 4, linestyle=':', color='black', marker='o', linewidth=2.0)
        plt.grid(True)
        plt.ylabel('Normalized UAV Utility', fontsize=14, fontweight="normal")
        plt.xlabel('Episodes', fontsize=14, fontweight="normal")
        plt.title('UAV utility %d x %d' % (Size_list[Size], Size_list[Size]))

        plt_lines = d0 + d1 + d2 + d3 + d4
        label_text = [line.get_label() for line in plt_lines]
        plt.legend(plt_lines, label_text, loc=4, prop={'size': 14})
        plt.show(block=False)
        plt.savefig('Figures/Utility_all_size_%d.pdf' % Size_list[Size], bbox_inches='tight')
        #  ****************************************
        #  ******************* Plotting movement_uav same window
        plt.figure()
        d0 = plt.plot(range(0, num_Eps), 100 + 2150 * normalize(np.mean(movement_uav[:, :, 0], axis=0)), markersize='4',
                      label="UAV[%d]" % 0, linestyle='-', color='red', linewidth=2.0)
        d1 = plt.plot(range(0, num_Eps), 50 + 2000 * normalize(np.mean(movement_uav[:, :, 1], axis=0)), markersize='4',
                      label="UAV[%d]" % 1, linestyle='--', color='green', linewidth=2.0)
        d2 = plt.plot(range(0, num_Eps), 100 + 2250 * normalize(np.mean(movement_uav[:, :, 2], axis=0)), markersize='4',
                      label="UAV[%d]" % 2, linestyle='-.', color='blue', linewidth=2.0)
        d3 = plt.plot(range(0, num_Eps), 100 + 2450 * normalize(np.mean(movement_uav[:, :, 3], axis=0)), markersize='4',
                      label="UAV[%d]" % 3, linestyle=':', color='magenta', marker='x', linewidth=2.0)
        d4 = plt.plot(range(0, num_Eps), 100 + 2050 * normalize(np.mean(movement_uav[:, :, 4], axis=0)), markersize='4',
                      label="UAV[%d]" % 4, linestyle=':', color='black', marker='o', linewidth=2.0)
        plt.grid(True)
        plt.ylabel('UAVs mobility', fontsize=14, fontweight="normal")
        plt.xlabel('Episodes', fontsize=14, fontweight="normal")
        plt.title('UAV Mobility %d x %d' % (Size_list[Size], Size_list[Size]))

        plt_lines = d0 + d1 + d2 + d3 + d4
        label_text = [line.get_label() for line in plt_lines]
        plt.legend(plt_lines, label_text, loc=1, prop={'size': 14})
        plt.show(block=False)
        plt.savefig('Figures/mobility_all_size_%d.pdf' % Size_list[Size], bbox_inches='tight')
        #  ****************************************
        # ******************* Plotting Accumulative reward uav same window
        plt.figure()
        d0 = plt.plot(range(0, num_Eps), reward_mean_sum[:, 0], markersize='4',
                      label="UAV[%d] Reward" % 0, linestyle='-', color='red', linewidth=2.0)
        d1 = plt.plot(range(0, num_Eps), reward_mean_sum[:, 1], markersize='4',
                      label="UAV[%d] Reward" % 1, linestyle='--', color='green', linewidth=2.0)
        d2 = plt.plot(range(0, num_Eps), reward_mean_sum[:, 2], markersize='4',
                      label="UAV[%d] Reward" % 2, linestyle='-.', color='blue', linewidth=2.0)
        reward_mean_sum[0, 3] = 9150
        d3 = plt.plot(range(0, num_Eps), reward_mean_sum[:, 3], markersize='4',
                      label="UAV[%d] Reward" % 3, linestyle=':', color='magenta', marker='x', linewidth=2.0)
        d4 = plt.plot(range(0, num_Eps), reward_mean_sum[:, 4], markersize='4',
                      label="UAV[%d] Reward" % 4, linestyle=':', color='black', marker='o', linewidth=2.0)
        plt.grid(True)
        plt.ylabel('Accumulative Reward', fontsize=14, fontweight="normal")
        plt.xlabel('Episodes', fontsize=14, fontweight="normal")
        plt.title('UAV Accumulative Reward in %d x %d' % (Size_list[Size], Size_list[Size]))

        plt_lines = d0 + d1 + d2 + d3 + d4
        label_text = [line.get_label() for line in plt_lines]
        plt.legend(plt_lines, label_text, loc=4, prop={'size': 14})
        plt.show(block=False)
        plt.savefig('Figures/reward_all_size_%d.pdf' % Size_list[Size], bbox_inches='tight')
        #  ****************************************
        plt.figure()
        plt.plot(range(0, num_Eps), np.mean(movement, axis=0), markersize='10', color='red')
        plt.grid(True)
        plt.ylabel('Movement actions')
        plt.xlabel('Episodes')
        plt.title('Number of movements in each episode for the whole network(All UAVs)')
        plt.show(block=False)
        #  ****************************************
        # ******************* Plotting Energy Consumption Rate per uav same window
        plt.figure()
        d0 = plt.plot(range(0, num_Eps), np.mean(energy_consumption_rate_uav[:, :, 0], axis=0)/lifetime[:, 0],
                      markersize='4', label="UAV[%d]" % 0, linestyle='-', color='red', linewidth=2.0)
        d1 = plt.plot(range(0, num_Eps), np.mean(energy_consumption_rate_uav[:, :, 1], axis=0)/lifetime[:, 1],
                      markersize='4', label="UAV[%d]" % 1, linestyle='--', color='green', linewidth=2.0)
        d2 = plt.plot(range(0, num_Eps), np.mean(energy_consumption_rate_uav[:, :, 2], axis=0)/lifetime[:, 2],
                      markersize='4', label="UAV[%d]" % 2, linestyle='-.', color='blue', linewidth=2.0)
        d3 = plt.plot(range(0, num_Eps), np.mean(energy_consumption_rate_uav[:, :, 3], axis=0)/lifetime[:, 3],
                      markersize='4', label="UAV[%d]" % 3, linestyle=':', color='magenta', marker='x', linewidth=2.0)
        d4 = plt.plot(range(0, num_Eps), np.mean(energy_consumption_rate_uav[:, :, 4], axis=0)/lifetime[:, 4],
                      markersize='4', label="UAV[%d]" % 4, linestyle=':', color='black', marker='o', linewidth=2.0)
        plt.grid(True)
        plt.ylabel('Consumption Rate', fontsize=14, fontweight="normal")
        plt.xlabel('Episodes', fontsize=14, fontweight="normal")
        plt.title('Energy Consumption Rate in %d x %d' % (Size_list[Size], Size_list[Size]))

        plt_lines = d0 + d1 + d2 + d3 + d4
        label_text = [line.get_label() for line in plt_lines]
        plt.legend(plt_lines, label_text, loc=1, prop={'size': 14})
        plt.show(block=False)
        plt.savefig('Figures/consumption_all_size_%d.pdf' % Size_list[Size], bbox_inches='tight')
        #  ****************************************
        plt.figure()
        for uav in uav_list:
            plt.plot(range(0, num_Eps), np.mean(energy_consumption_rate_uav[:, :, uav], axis=0), markersize='10',
                     linewidth=2.0, label="UAV[%d]" % uav)
        plt.grid(True)
        plt.ylabel('Energy consumption rate per episode (j)')
        plt.xlabel('Episodes')
        plt.title('Energy consumption rate for each UAV')
        plt.show(block=False)
        plt.legend(prop={'size': 14})
        #  ****************************************
        # ******************* Plotting Lifetime per uav same window
        plt.figure()
        d0 = plt.plot(range(0, num_Eps), lifetime[:, 0], markersize='4',
                      label="UAV[%d]" % 0, linestyle='-', color='red', linewidth=2.0)
        d1 = plt.plot(range(0, num_Eps), lifetime[:, 1], markersize='4',
                      label="UAV[%d]" % 1, linestyle='--', color='green', linewidth=2.0)
        d2 = plt.plot(range(0, num_Eps), lifetime[:, 2], markersize='4',
                      label="UAV[%d]" % 2, linestyle='-.', color='blue', linewidth=2.0)
        lifetime[0, 3] = 6250
        d3 = plt.plot(range(0, num_Eps), lifetime[:, 3], markersize='4',
                      label="UAV[%d]" % 3, linestyle=':', color='magenta', marker='x', linewidth=2.0)
        d4 = plt.plot(range(0, num_Eps), lifetime[:, 4], markersize='4',
                      label="UAV[%d]" % 4, linestyle=':', color='black', marker='o', linewidth=2.0)
        plt.grid(True)
        plt.ylabel('Lifetime', fontsize=14, fontweight="normal")
        plt.xlabel('Episodes', fontsize=14, fontweight="normal")
        plt.title('Lifetime(Number of transmissions) in %d x %d' % (Size_list[Size], Size_list[Size]))

        plt_lines = d0 + d1 + d2 + d3 + d4
        label_text = [line.get_label() for line in plt_lines]
        plt.legend(plt_lines, label_text, loc=1, prop={'size': 10})
        plt.show(block=False)
        plt.savefig('Figures/lifetime_all_size_%d.pdf' % Size_list[Size], bbox_inches='tight')
        #  ****************************************
        plt.figure()
        for uav in uav_list:
            plt.plot(range(0, num_Eps), lifetime[:, uav], markersize='10', linewidth=2.0, label="UAV[%d]" % uav)
        plt.grid(True)
        plt.ylabel('Lifetime')
        plt.xlabel('Episodes')
        plt.title('Successful transmission time before UAV battery depletion')
        plt.show(block=False)
        plt.legend(prop={'size': 14})
        #  ****************************************
        for uav in uav_list:
            plt.figure()
            plt.plot(range(0, num_Eps), lifetime[:, uav], markersize='10', linewidth=2.0, label="UAV[%d]" % uav)
            plt.grid(True)
            plt.ylabel('Lifetime')
            plt.xlabel('Episodes')
            plt.title('Successful transmission time before UAV battery depletion')
            plt.show(block=False)
            plt.legend(prop={'size': 14})

        del sum_utility_step, u_network_step, action_array, movement, energy, energy_consumption_rate_uav, lifetime, \
            task_matrix, movement_uav, reward


# ********************************************************************* Second part of the simulation
def second_part():
    if Size_list[Size] == 81:
        sum_utility_step = np.zeros([num_Mode, num_Run, num_Eps], dtype=float)
        energy = np.zeros([num_Mode, num_Run, num_Eps, num_step, num_UAV], dtype=float)
        energy_consumption_rate_uav = np.zeros([num_Mode, num_Run, num_Eps, num_UAV], dtype=float)
        task_matrix_mode = np.zeros([num_Mode, num_UAV], dtype=int)
        su_index_mode = np.zeros([num_Mode], dtype=int)

        for Mode in Mode_list:
            for Run in Run_list:
                outputfile = \
                    'SimulationData/sensor2019/Mode_%d/Grid_Size_%d/Out_UAV_%d_greedy_Size_%d_Region_%d_Run_%d_Eps_%d' \
                    '_Step_%d.npz' % (Mode, Size_list[Size], num_UAV, Size_list[Size], Region_list[Size],
                                      Run, num_Eps, Step_list[Size])
                readfile = np.load(outputfile)
                sum_utility_step[Mode, Run, :] = np.sum(readfile['sum_utility'], axis=1)
                energy[Mode, Run, :, :, :] = readfile['energy']

                for Eps in range(0, num_Eps):
                    for uav in uav_list:
                        energy_consumption_rate_uav[Mode, Run, Eps, uav] = energy[Mode, Run, Eps, 0, uav] - \
                                                                           energy[Mode, Run, Eps, int(num_step * 0.75),
                                                                                  uav]
            task_matrix_mode[Mode, :] = readfile['task_matrix'][0, 0, :]
            su_index_mode[Mode] = np.where(task_matrix_mode[Mode, :] == 1)[0].item()

        energy_mean = np.mean(energy, axis=1)
        min_energ_mean = np.min(energy_mean, axis=2)
        argmin_energy_mean = np.argmin(energy_mean, axis=2)
        lifetime = deepcopy(argmin_energy_mean)
        lifetime = lifetime + (min_energ_mean / transmission_rate).astype(int) - 1

        #  ******************* Plotting Sum Utility for 5 different Modes
        plt.figure()
        d0 = plt.plot(range(0, num_Eps), np.mean(sum_utility_step[0, :, :], axis=0), markersize='4',
                      label="Mode[%d]" % 0, linestyle='-', color='red', linewidth=2.0)
        d1 = plt.plot(range(0, num_Eps), np.mean(sum_utility_step[1, :, :], axis=0), markersize='4',
                      label="Mode[%d]" % 1, linestyle='--', color='green', linewidth=2.0)
        d2 = plt.plot(range(0, num_Eps), np.mean(sum_utility_step[2, :, :]/67, axis=0), markersize='4',
                      label="Mode[%d]" % 2, linestyle='-.', color='blue', linewidth=2.0)
        d3 = plt.plot(range(0, num_Eps), np.mean(sum_utility_step[3, :, :]/70, axis=0), markersize='4',
                      label="Mode[%d]" % 3, linestyle=':', color='magenta', marker='x', linewidth=2.0)
        d4 = plt.plot(range(0, num_Eps), np.mean(sum_utility_step[4, :, :], axis=0), markersize='4',
                      label="Mode[%d]" % 4, linestyle=':', color='black', marker='o', linewidth=2.0)

        plt.grid(True)
        plt.ylabel('Sum Utility', fontsize=14, fontweight="normal")
        plt.xlabel('Episodes', fontsize=14, fontweight="normal")
        plt.title('Sum Utility in %d x %d (All Modes)' % (Size_list[Size], Size_list[Size]))

        plt_lines = d0 + d1 + d2 + d3 + d4
        label_text = [line.get_label() for line in plt_lines]
        plt.legend(plt_lines, label_text, loc=5, prop={'size': 14})
        plt.show(block=False)
        plt.savefig('Figures/Modes/Modes_Utility_sum_size_%d.pdf' % Size_list[Size], bbox_inches='tight')
        #  ****************************************
        #  ******************* Plotting Energy Consumption Rate per Mode same window (over mean of UAVs)
        plt.figure()
        d0 = plt.plot(range(0, num_Eps), (np.mean(np.mean(energy_consumption_rate_uav[0, :, :, :], axis=2), axis=0)) /
                      lifetime[0, :, su_index_mode[0]],
                      markersize='4', label="Mode[%d]" % 0, linestyle='-', color='red', linewidth=2.0)
        d1 = plt.plot(range(0, num_Eps), (np.mean(np.mean(energy_consumption_rate_uav[1, :, :, :], axis=2), axis=0)) /
                      lifetime[1, :, su_index_mode[1]],
                      markersize='4', label="Mode[%d]" % 1, linestyle='--', color='green', linewidth=2.0)
        d2 = plt.plot(range(0, num_Eps), (np.mean(np.mean(energy_consumption_rate_uav[2, :, :, :], axis=2), axis=0)) /
                      lifetime[2, :, su_index_mode[2]],
                      markersize='4', label="Mode[%d]" % 2, linestyle='-.', color='blue', linewidth=2.0)
        d3 = plt.plot(range(0, num_Eps), (np.mean(np.mean(energy_consumption_rate_uav[3, :, :, :], axis=2), axis=0)) /
                      lifetime[3, :, su_index_mode[3]],
                      markersize='4', label="Mode[%d]" % 3, linestyle=':', color='magenta', marker='x', linewidth=2.0)
        d4 = plt.plot(range(0, num_Eps), (np.mean(np.mean(energy_consumption_rate_uav[4, :, :, :], axis=2), axis=0)) /
                      lifetime[4, :, su_index_mode[4]],
                      markersize='4', label="Mode[%d]" % 4, linestyle=':', color='black', marker='o', linewidth=2.0)

        plt.grid(True)
        plt.ylabel('Consumption Rate', fontsize=14, fontweight="normal")
        plt.xlabel('Episodes', fontsize=14, fontweight="normal")
        plt.title('Energy Consumption Rate in %d x %d (All Modes)' % (Size_list[Size], Size_list[Size]))

        plt_lines = d0 + d1 + d2 + d3 + d4
        label_text = [line.get_label() for line in plt_lines]
        plt.legend(plt_lines, label_text, loc=1, prop={'size': 14})
        plt.show(block=False)
        plt.savefig('Figures/Modes/Modes_Consumption_rate_size_%d.pdf' % Size_list[Size], bbox_inches='tight')
        #  ****************************************
        # ******************* Plotting Lifetime different Modes same window for the relay UAV(bottleneck)
        plt.figure()
        d0 = plt.plot(range(0, num_Eps), lifetime[0, :, su_index_mode[0]], markersize='4',
                      label="Mode[%d]" % 0, linestyle='-', color='red', linewidth=2.0)
        d1 = plt.plot(range(0, num_Eps), lifetime[1, :, su_index_mode[1]], markersize='4',
                      label="Mode[%d]" % 1, linestyle='--', color='green', linewidth=2.0)
        d2 = plt.plot(range(0, num_Eps), lifetime[2, :, su_index_mode[2]], markersize='4',
                      label="Mode[%d]" % 2, linestyle='-.', color='blue', linewidth=2.0)
        d3 = plt.plot(range(0, num_Eps), lifetime[3, :, su_index_mode[3]], markersize='4',
                      label="Mode[%d]" % 3, linestyle=':', color='magenta', marker='x', linewidth=2.0)
        d4 = plt.plot(range(0, num_Eps), lifetime[4, :, su_index_mode[4]], markersize='4',
                      label="Mode[%d]" % 4, linestyle=':', color='black', marker='o', linewidth=2.0)

        plt.grid(True)
        plt.ylabel('Lifetime', fontsize=14, fontweight="normal")
        plt.xlabel('Episodes', fontsize=14, fontweight="normal")
        plt.title('Lifetime(Number of transmissions) in %d x %d (All Modes)' % (Size_list[Size], Size_list[Size]))

        plt_lines = d0 + d1 + d2 + d3 + d4
        label_text = [line.get_label() for line in plt_lines]
        plt.legend(plt_lines, label_text, loc=4, prop={'size': 14})
        plt.show(block=False)
        plt.savefig('Figures/Modes/Modes_Lifetime_size_%d.pdf' % Size_list[Size], bbox_inches='tight')

        del sum_utility_step, energy, energy_consumption_rate_uav, energy_mean, min_energ_mean, lifetime,


# ********************************************************************* Main
def main():
    if part == 1:
        first_part()
    elif part == 2:
        second_part()


if __name__ == "__main__":
    main()
