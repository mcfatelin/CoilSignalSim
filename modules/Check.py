####    this is a test code to check if the response function is right 
####    and to deepen my understanding of fft
####    by Beige Liu @20220513

import numpy as np
from  scipy import fftpack


#######initialize constant####
T2              =  0.004  #in s from chang chen's ppt
consider_freq   = 40000   #40000 #in Hz
Polarization    = 0.8     #from what Doc.Jiang said about P
sampling_step   = 1e-6    #in s
sampling_number = 1000    #from Doc.Lin 's code
theta0          = np.pi/3 

def _Response_funtion(freq):
        ### calculate the complex response value 
        gama   = 1/(T2*2*np.pi)
        output = \
            Polarization/(2*np.pi)*np.exp(1j*theta0)*(gama - 1j*(freq-consider_freq))/(gama**2 + (freq - consider_freq)**2)+\
            Polarization/(2*np.pi)*np.exp(-1j*theta0)*(gama - 1j*(freq+consider_freq))/(gama**2 + (freq + consider_freq)**2)
        return output
def _transfer_based_on_voltage(freq,voltage,phase):
        ## first fft voltage
        spec     = fftpack.fft(voltage)
        remain   = np.copy(spec)
        ## calculate response value
        response = _Response_funtion(freq)
        ## multiply the response value to spec
        spec     = np.multiply(
            spec,
            response
            )
        ## inverse fft spec
        voltage_Mag = fftpack.ifft(spec)
        return voltage_Mag,response,remain

########
###check
########



########initialize signal##########
freq      = np.linspace(37000,43000,10000)  ### mix frequency between 37~43kHz
voltage   = np.zeros(freq.shape[0])
amp       = np.random.random((freq.shape[0]))### random amplitude
phase1    = 2*np.pi*np.random.random((freq.shape[0]))# random phase
signal_length     = 1e-3                     ### in s
simulation_sample = 10000
t         = np.linspace(0,signal_length,simulation_sample) #in s
for i in range(freq.shape[0]):
    voltage += amp[i]*np.sin(2*np.pi*t*freq[i]+phase1[i])



#########sampling###################
step          = int(sampling_step/(signal_length/simulation_sample))
input_voltage = np.zeros(sampling_number,dtype = float)
phase         = np.zeros(sampling_number,dtype = float)
sample_t      = np.zeros(sampling_number,dtype = float)
for i in range(sampling_number):
    input_voltage[i]    = voltage[int(i*step)]
    phase[i]            = phase1[int(i*step)]
    sample_t[i]         = t[int(step*i)]
print(sample_t.shape[0])




#########fft part######

######get fft frequency
freq             = fftpack.fftfreq(
                                   sampling_number,
                                   sampling_step)
######first fft voltage
spec             = fftpack.fft(input_voltage)
######get the response
response         = _Response_funtion(freq)
######multiply response with spectrum
output_spec      = np.multiply(spec,
                               response)
######inverse fft spectrum
output_voltage   = fftpack.ifft(output_spec)

######calculate spectrum power to plot
output_spec      = abs(output_spec)**2
spec             = abs(spec)**2
response         = abs(response)**2






############plot part###########
import matplotlib.pyplot as plt
from matplotlib import gridspec

########### plot input&output spectrum
plt.figure(figsize=(10,7),dpi = 100)
plt.plot(   freq[:],
            output_spec[:],
            color = 'r',
            #s = 1,
            label = 'output_spectrum')
plt.plot(   freq[:],
            5*1e-6*spec[:],
            color = 'b',
            #s =1,
            label = '5*1e-6*original_spectrum')
plt.xlabel('f(Hz)',fontsize=18)
plt.ylabel('power',fontsize=18)
plt.legend(fontsize=18)
plt.tick_params(labelsize=18)
plt.title('consider frequency = 40kHz; T2 = 0.004 '+'\n'+';random phase;random ampitiude',fontsize=18)
plt.show()
plt.savefig(r'C:\Users\admin\Desktop\spectrum_random_a_p'+format('.png'))

########### plot input&output signal wave
plt.figure(figsize=(10,7),dpi = 100)
plt.scatter(sample_t,
            1e-2*input_voltage,
            color = 'r',
            s = 1,
            label = '0.01*input signal')
plt.scatter(sample_t,
            output_voltage,
            color = 'b',
            s = 1,
            label = 'output signal')
plt.xlabel('t(s)',fontsize=18)
plt.ylabel('U(V)',fontsize=18)
plt.legend(fontsize=18)
plt.title('consider frequency = 40kHz; T2 = 0.004;'+'\n'+'random phase;random ampitiude',fontsize=18)
plt.tick_params(labelsize=18)
plt.show()
plt.savefig(r'C:\Users\admin\Desktop\Magnetometer_signal_shaping_zero_a_p'+format('.png'))


#######problems that cannot be understood:
#######1.为什么在原始信号spectrum会有一个峰值,无论在加随机相位还是不加随机相位情况下，均如此 -> 信号集中分布在37~43kHz区间，可以理解
#######2.如何理解fftpack.fftfreq返回值中的负频率 -> 左行右行信号的体现，可以取绝对值再平均值
#######3.如何理解input signal比output signal大那么多的情况.能量不守恒，why? -> 选频之后能量本来就不守恒吧