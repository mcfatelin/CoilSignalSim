import numpy as np
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
import scipy.io as scio
import scipy.special as spl
from PIL import Image
import sympy

'''
1、输入信号部分
输入信号存在时域解析解
输入信号存在频域解析解
直接取离散的时域信号，dft成频域，得到的结果与输入信号的频域解析解存在差异，这是由时域截取时间窗不同导致的
故：直接取输入信号的频域解析解，离散，代入仿真
'''

## 导入文件，输入波形 from Kang
path = r'D:\data.mat'
data = scio.loadmat(path)
t = data['Expression1'][:, 0]
amp = data['Expression1'][:, 1]
plt.figure()
plt.plot(t, amp)
plt.xlabel('t(s)')
plt.ylabel('Voltage amplitude(V)')
plt.suptitle('input coil')
L = len(t)
Fs = 1 / (t[2] - t[1])  # 采样率
f = np.array([Fs / L * i for i in range(int((L + 1) / 2))])
## 输入波形 FFT
Y = fft(amp)
Y_input = Y
P2 = np.abs(Y / L)
P1 = P2[0:int((L + 1) / 2):1]
P1[1::] = 2 * P1[1::]
sig_f = P1
dc = P1[0]
plt.figure()
plt.loglog(f, P1)
plt.xlabel('f(Hz)')
plt.ylabel('Voltage amplitude(V)')
plt.suptitle('input coil')
'''
2、传递函数部分
输入:
    f：离散频域数组
    circuit_form：电路形式{
        1：  无谐振电路
        2：  串联谐振电路形式1
        3：  并联谐振电路形式2
        4：  并联谐振电路形式3
        5：  并联谐振电路形式4
        6：  并联谐振电路形式5
        7：  并联谐振电路形式6
    }
    输入线圈参数{
        T1：     输入线圈环境温度
        Lin：    输入线圈感值
        Cin：    输入线圈容值
        Rin：    输入线圈阻值
        r_A1：   交流电阻阻值随频率变化斜率（直线拟合），ohm/Hz
    }
    外接电感电容参数{
        T2：     外接电感电容环境温度
        Lout：   外接电感感值
        Cout：   外接电感寄生电容容值
        Rout：   外接电感电阻阻值
        r_A2：   交流电阻随频率变化斜率（直线拟合），ohm/Hz
        C：      外接电容容值
    }    
    磁场转换线圈参数{
        T3：     磁场转换线圈环境温度
        Lload：  磁场转换线圈感值
        Cload：  磁场转换线圈容值
        Rload：  磁场转换线圈阻值
        r_A3：   交流电阻随频率变化斜率（直线拟合），ohm/Hz
    }        
输出：
    Dict：{'signal': {'f': f, 'respond': respond, 'phase_x': phase_x, 'phase_y': phase_y},
        'noise': {'f': f, 'respond': respond, 'phase_x': phase_x, 'phase_y': phase_y}}
    circuit_pic：{
        pic1：  无谐振电路
        pic2：  串联谐振电路形式1
        pic3：  并联谐振电路形式2
        pic4：  并联谐振电路形式3
        pic5：  并联谐振电路形式4
        pic6：  并联谐振电路形式5
        pic7：  并联谐振电路形式6
    }
'''

f = f
circuit_form = 0  # 无外接电感电容的模型


# 给出相关的电路图
def circuit_picshow(num):
    picshow = {
        0: Image.open('D:\pythonProject\pic0.png'),
        1: Image.open('D:\pythonProject\pic1.png'),
        2: Image.open('D:\pythonProject\pic2.png'),
        3: Image.open('D:\pythonProject\pic3.png'),
        4: Image.open('D:\pythonProject\pic4.png'),
        5: Image.open('D:\pythonProject\pic5.png'),
        6: Image.open('D:\pythonProject\pic6.png'),
        7: Image.open('D:\pythonProject\pic7.png'),
    }
    return picshow.get(num, None)


img = circuit_picshow(circuit_form)
img.show()

# 输入线圈参数，根据文献
T1 = 78  # 温度
r_A1 = 9.5e-3  # 交流电阻随频率变化曲线
Lin = 2.313  # 输入线圈等效电感
Cin = 42.1e-6  # 输入线圈等效电容//-12
Rin = 158.3  # 输入线圈等效电阻

# 磁场转换线圈参数，实验测得的结果
T3 = 298  # 温度
r_A3 = 1e-6  # 交流电阻随频率变化曲线
Lload = 1e-4  # 磁场转换线圈等效电感
Cload = 1e-10  # 磁场转换线圈等效电容
Rload = 1e-1  # 磁场转换线圈等效电阻


# 并联函数定义
def para(a, b):
    result = 1 / (1 / a + 1 / b)
    return result


# 传递函数1
def trans_sig(f):
    result = 1 / (1j * 2 * np.pi * f * Lin + (r_A1 * f + 1) * Rin + para(1 / (1j * 2 * np.pi * f * (Cin + Cload)),
                                                                         1j * 2 * np.pi * f * Lload + (
                                                                                 r_A3 * f + 1) * Rload)) * (
                     1 / (1j * 2 * np.pi * f * (Cin + Cload))) / (
                     1 / (1j * 2 * np.pi * f * (Cin + Cload)) + 1j * 2 * np.pi * f * Lload + (r_A3 * f + 1) * Rload)
    return result


# 传递函数3
def trans_noise(f):
    result = 1 / (1j * 2 * np.pi * f * Lload + (r_A3 * f + 1) * Rload + para(1 / (1j * 2 * np.pi * f * (Cin + Cload)),
                                                                             1j * 2 * np.pi * f * Lin + (
                                                                                     r_A1 * f + 1) * Rin))
    return result


respond_sig = [trans_sig(f) for f in f[1::]]  # f[0]是dc分量，传递函数存在1/f 不能直接代入计算
temp = np.conj(respond_sig[1::])
temp = temp[::-1]
# respond_sig=np.append(respond_sig, temp)
amp_sig = np.abs(respond_sig)
phase_x_sig = np.real(respond_sig)
phase_y_sig = np.imag(respond_sig)

respond_noise = [trans_noise(f) for f in f[1::]]
temp = np.conj(respond_noise[1::])
temp = temp[::-1]
# respond_noise=np.append(respond_noise, temp)
amp_noise = np.abs(respond_noise)
phase_x_noise = np.real(respond_noise)
phase_y_noise = np.imag(respond_noise)

Dict = {
    'signal': {
        'f': f,
        'amp': amp_sig,
        'phase_x': phase_x_sig,
        'phase_y': phase_y_sig
    },
    'noise': {
        'f': f,
        'amp': amp_noise,
        'phase_x': phase_x_noise,
        'phase_y': phase_y_noise
    },
    'parameter': {
        'T1': 78,  # 输入线圈温度
        'r_A1': 9.5e-3,  # 交流电阻随频率变化曲线
        'Lin': 2.313,  # 输入线圈等效电感
        'Cin': 42.1e-12,  # 输入线圈等效电容
        'Rin': 158.3,  # 输入线圈等效电阻
        'T3': 298,  # 磁场转换线圈温度
        'r_A3': 1e-6,  # 交流电阻随频率变化曲线
        'Lload': 1e-4,  # 磁场转换线圈等效电感
        'Cload': 1e-10,  # 磁场转换线圈等效电容
        'Rload': 1e-1  # 磁场转换线圈等效电阻

    }
}

'''
3、输出信号与输出噪声求解
'''
# 输出信号求解
output_sig = np.multiply(sig_f[1::], respond_sig)
output_sig_dc = np.append(dc / (Rin + Rload), output_sig)  # 补上直流分量

output_f = output_sig_dc * len(output_sig_dc)
temp = np.conj(output_f[1::])
temp = temp[::-1]
output_f = np.append(output_f, temp)
output = ifft(output_f)
plt.figure()
plt.plot(t, np.real(output))
plt.xlabel('t(s)')
plt.ylabel('current amplitude(A)')
plt.suptitle('output current')

# 输出噪声求解
k_bol = 1.380649 * 10 ** -23  # J/K
phase1 = 2 * np.pi * np.random.rand(len(f))
phase3 = 2 * np.pi * np.random.rand(len(f))
vn1 = np.multiply(np.sqrt(4 * k_bol * T1 * (r_A1 * f + 1) * Rin), np.exp(1j * phase1))
vn3 = np.multiply(np.sqrt(4 * k_bol * T3 * (r_A3 * f + 1) * Rload), np.exp(1j * phase3))
outputn_noise1 = np.multiply(vn1[1::], respond_sig)
outputn_noise3 = np.multiply(vn3[1::], respond_noise)

outputn_noise1_dc = np.append(0, outputn_noise1)  # 补上直流分量
outputn_noise3_dc = np.append(0, outputn_noise3)  # 补上直流分量

output_f = (np.array(output_sig_dc) + np.array(outputn_noise1_dc) + np.array(outputn_noise3_dc)) * len(output_sig_dc)
temp = np.conj(output_f[1::])
temp = temp[::-1]
output_f = np.append(output_f, temp)
output = ifft(output_f)
plt.figure()
plt.plot(t, np.real(output))
plt.xlabel('t(s)')
plt.ylabel('current amplitude(A)')
plt.suptitle('output current')
plt.show()
