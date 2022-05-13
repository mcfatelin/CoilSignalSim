from MagnetometerCalculator import MagnetometerCalculator
import numpy as np
import pickle as pkl
config_filename = r'C:\Users\admin\Desktop\Magneto_config'
Dict     = {'consider_freq':40000,
            'T2': 0.002,
            'Polarization':0.8,
            'theta0':np.pi/3}
pkl.dump(Dict,open(config_filename,'wb+'))

########initialize signal##########
freq      = np.linspace(37000,43000,10000)  ### mix frequency between 37~43kHz
voltage   = np.zeros(freq.shape[0])
noise     = np.zeros(freq.shape[0])
amp       = np.random.random((freq.shape[0]))### random amplitude
phase1    = 2*np.pi*np.random.random((freq.shape[0]))# random phase
signal_length     = 1e-3                     ### in s
simulation_sample = 10000
t         = np.linspace(0,signal_length,simulation_sample) #in s
for i in range(freq.shape[0]):
    voltage += amp[i]*np.sin(2*np.pi*t*freq[i]+phase1[i])
    noise   += np.cos(2*np.pi*t*freq[i]*10+phase1[i])
#########sampling###################
sampling_step   = 1e-6    #in s
sampling_number = 1000    #from Doc.Lin 's code
step          = int(sampling_step/(signal_length/simulation_sample))
input_voltage = np.zeros(sampling_number,dtype = float)
input_noise   = np.zeros(sampling_number,dtype = float)
sample_t      = np.zeros(sampling_number,dtype = float)
for i in range(sampling_number):
    input_voltage[i]    = voltage[int(i*step)]
    sample_t[i]         = t[int(step*i)]
    input_noise[i]      = noise[int(step*i)]

RLC_filename       = r'C:\Users\admin\Desktop\RLC_signal'
Dict               = {}
Dict['voltages_RLC']= input_voltage
Dict['noises_RLC']  = input_noise
f                  = open(RLC_filename,'w')
f.write(str(Dict))
f.close()

config_file      = {'filename':config_filename}
Calc     = MagnetometerCalculator(**{'filename':config_filename})
Obj      = Calc.calculateVoltages(**Dict)
signal   = Obj[0]
noise    = Obj[1]
############plot part###########
import matplotlib.pyplot as plt
from matplotlib import gridspec
########### plot input&output signal wave
plt.figure(figsize=(10,7),dpi = 100)
plt.scatter(sample_t,
            noise,
            color = 'r',
            s = 1,
            label = 'high frequency noise')
plt.scatter(sample_t,
            signal,
            color = 'b',
            s = 1,
            label = 'output signal')
plt.xlabel('t(s)',fontsize=18)
plt.ylabel('U(V)',fontsize=18)
plt.legend(fontsize=18)
plt.title('consider frequency = 40kHz; T2 = 0.004;'+'\n'+'random phase;random ampitiude',fontsize=18)
plt.tick_params(labelsize=18)
plt.show()
plt.savefig(r'C:\Users\admin\Desktop\Magnetometer_signal'+format('.png'))
