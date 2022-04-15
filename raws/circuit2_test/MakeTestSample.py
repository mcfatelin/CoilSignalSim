###########################################
## This code is for making a test circuit
###########################################
import numpy as np
import scipy.io as scio

from copy import deepcopy


############################
## Some of the functions
#############################

## 谐振电路参数
# 输入线圈参数
T1 = 78  # 温度
r_A1 = 9.5e-3  # 交流电阻随频率变化曲线
Lin = 2.313  # 输入线圈等效电感
Cin = 42.1e-6  # 输入线圈等效电容
Rin = 158.3  # 输入线圈等效电阻
# 外接电感和电容参数
T2 = 298  # 温度
r_A2 = 0  # 交流电阻随频率变化曲线
Lout = 1e-3  # 外接电感
Cout = 1e-6  # 外接电感等效电容部分
Rout = 1e-6  # 外接电感等效电阻部分
C = 1e-6  # 外接电容
# 磁场转换线圈参数
T3 = 298  # 温度
r_A3 = 1e-6  # 交流电阻随频率变化曲线
Lload = 1e-4  # 磁场转换线圈等效电感
Cload = 1e-10  # 磁场转换线圈等效电容
Rload = 1.0e-1  # 磁场转换线圈等效电阻


def para(a, b):
    result = 1 / (1 / a + 1 / b)
    return result


# response function for signal coil and coil resistor
def trans1(f):
    result = 1 / (1j * 2 * np.pi * f * Lin + (r_A1 * f + 1) * Rin + para(1 / (1j * 2 * np.pi * f * (Cin + Cload)),
                                                                         1j * 2 * np.pi * f * Lload + (
                                                                                 r_A3 * f + 1) * Rload)) * (
                     1 / (1j * 2 * np.pi * f * (Cin + Cload))) / (
                     1 / (1j * 2 * np.pi * f * (Cin + Cload)) + 1j * 2 * np.pi * f * Lload + (r_A3 * f + 1) * Rload)
    return result


# response function for induction coil resistor
def trans3(f):
    result = 1 / (1j * 2 * np.pi * f * Lload + (r_A3 * f + 1) * Rload + para(1 / (1j * 2 * np.pi * f * (Cin + Cload)),
                                                                             1j * 2 * np.pi * f * Lin + (
                                                                                     r_A1 * f + 1) * Rin))
    return result



############################
## Make the circuit parameters
#############################
OutDict                     = {}

freq                        = np.logspace(
    np.log10(1),
    np.log10(4e6),
    100001,
)


response1                   = trans1(freq)
response2                   = trans3(freq)

###########
# Signal
###########
OutDict['signal']           = {}

OutDict['signal']['f']              = freq*1e-6
OutDict['signal']['amp']            = np.abs(response1)
OutDict['signal']['phase_x']        = response1.real / np.abs(response1)
OutDict['signal']['phase_y']        = response1.imag / np.abs(response1)


###########
# Thermal noises
############
OutDict['noises']                   = []

OutDict['noises'].append({})
OutDict['noises'][0]['Rdc']         = Rin # Om
OutDict['noises'][0]['Rslope']      = r_A1*1e6 # /MHz
OutDict['noises'][0]['Temp']        = T1
OutDict['noises'][0]['f']           = freq*1e-6
OutDict['noises'][0]['amp']         = np.abs(response1)
OutDict['noises'][0]['phase_x']     = response1.real / np.abs(response1)
OutDict['noises'][0]['phase_y']     = response1.imag / np.abs(response1)

OutDict['noises'].append({})
OutDict['noises'][1]['Rdc']         = Rout # Om
OutDict['noises'][1]['Rslope']      = r_A3*1e6 # /MHz
OutDict['noises'][1]['Temp']        = T3
OutDict['noises'][1]['f']           = freq*1e-6
OutDict['noises'][1]['amp']         = np.abs(response2)
OutDict['noises'][1]['phase_x']     = response2.real / np.abs(response2)
OutDict['noises'][1]['phase_y']     = response2.imag / np.abs(response2)



############
## Output
#############

import pickle as pkl


pkl.dump(
    OutDict,
    open("circuit2_test.pkl", 'wb')
)
