"""
#################################
# Configuration File
#################################
"""

Config_General = {'NUM_UAV': 5, 'NUM_EPS': 40, 'NUM_STEP': 125, 'NUM_PKT': 1,
                  'Location_SaveFile': 0, 'CSI_SaveFile': 0, 'Energy_SaveFile': 0, 'PlotLocation': 1, 'DF': 0,
                  'printFlag': 1, 'PlotResult': 0, 'NUM_RUN': 20, 'SaveOutput': 1, 'Mode': 4}
#  ***** Modes ==>  0: Normal: Search primary region, search primary uav, search secondary regions + UAVs + RL
#  *****            1: Search Primary/Secondary regions + random UAV allocation + RL
#  *****            2: Random Region assignment + Random UAV Allocation + RL
#  *****            3: Random Region assignment + base-station UAV selection + RL
#  *****            4: Base-station Region assignment + Base-station UAV selection + Random actions(not RL)

Config_Param = {'T': 1, 'Noise': 1, 'Etha1': 1e-13, 'Etha2': 1, 'Etha3': 1, 'Gamma_punish1': 0.1, 'Gamma_punish2': 0.4,
                'lambda1': 2, 'lambda2': 2, 'lambda3': 0.4, 'lambda3_3': 0.1, 'Sigmoid_coef': 7}

Size = 32
Config_Dim = {'Height': 10, 'Length': Size, 'Width': Size, 'UAV_L_MAX': Size, 'UAV_L_MIN': 0, 'UAV_W_MIN': 0,
              'UAV_W_MAX': Size, 'Divider': Size, 'region': 64}

Config_Power = {'Power_fusion': 10, 'Power_source': 20, 'Power_pt': 10, 'Power_UAV_pr': 20, 'MaxEnergy': 5000.0,
                'MinEnergy': 4000.0, 'mob_consump_intra': 1.0, 'mob_consump_inter': 10.0, 'trans_consump': 0.5}

Config_RL = {'gamma': 0.3, 'alpha': 0.1, 'epsilon': 0.1, 'const_greedy': 0.9}

pathH = 'ConfigData/HMatrix_UAV_%d_Size_%d_Region_%d' % (Config_General.get('NUM_UAV'), Size, Config_Dim.get('region'))
pathDist = 'ConfigData/LocMatrix_UAV_%d_Size_%d_Region_%d' % (Config_General.get('NUM_UAV'), Size,
                                                              Config_Dim.get('region'))
pathEnergy = 'ConfigData/Energy_UAV_%d_Size_%d_Region_%d' % (Config_General.get('NUM_UAV'), Size,
                                                             Config_Dim.get('region'))
Config_Path = {'PathH': pathH, 'PathDist': pathDist, 'pathEnergy': pathEnergy}

Float_Precision = 2
