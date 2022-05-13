import numpy as np
from  scipy import fftpack
class _MagnetometerCalculator():
    def __init__(self):
        self._consider_freq = 40000
        self.T2             = 0.002#??
        self.Polarization       = 0.8
        return 0
    def _Response_funtion(self,freq,phase):
        ### calculate the complex response value 
        gama   = 1/(self.T2*2*np.pi)
        output = \
            self.Polarization/(2*np.pi)*np.exp(1j*phase)*(gama - 1j*(freq-self.consider_freq))/(gama**2 + (freq - self.consider_freq)**2)+\
            self.Polarization/(2*np.pi)*np.exp(-1j*phase)*(gama - 1j*(freq+self.consider_freq))/(gama**2 + (freq + self.consider_freq)**2)
        return output
    def _transfer_based_on_voltage(self,freq,voltage,amp,phase):
        ## first fft voltage
        spec     = fftpack.fft(voltage)
        ##calculate response value
        response = self._Response_funtion(freq,amp,phase)
        ## multiply the response value to spec
        spec     = np.multiply(
            spec,
            response
            )
        ## inverse fft spec
        voltage_Mag = fftpack.ifft(spec)
        return voltage_Mag