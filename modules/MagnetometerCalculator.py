### do not calculate the noise caused by magnetometer
import numpy as np
from  scipy import fftpack
import pickle as pkl

class MagnetometerCalculator():
    def __init__(self,**kwargs):
        if 'filename' not in kwargs.keys():
            raise ValueError(
                "Must give Magnetometer config filename!"
            )
        self._loadConfig(kwargs['filename'])
        return
    def _loadConfig(self,filename):
        ### load the magnetometer parameters
        Dict                    = pkl.load(open(filename,'rb'))
        self.consider_freq      = Dict['consider_freq']#40000  #in Hz
        self.T2                 = Dict['T2']           #in s
        self.Polarization       = Dict['Polarization'] #0.8
        self.magnetometer_phase = Dict['theta0']       #in arc
        return

    def _Response_funtion(self,freq):
        ### calculate the complex response value 
        gama   = 1/(self.T2*2*np.pi)
        output = \
            self.Polarization/(2*np.pi)*np.exp(1j*self.magnetometer_phase)*(gama - 1j*(freq-self.consider_freq))/(gama**2 + (freq - self.consider_freq)**2)+\
            self.Polarization/(2*np.pi)*np.exp(-1j*self.magnetometer_phase)*(gama - 1j*(freq+self.consider_freq))/(gama**2 + (freq + self.consider_freq)**2)
        return output

    def _transfer_based_on_voltages(self,freq,voltage):
        ## first fft voltage
        spec     = fftpack.fft(voltage)
        ##calculate response value
        response = self._Response_funtion(freq)
        ## multiply the response value to spec
        spec     = np.multiply(
            spec,
            response
            )
        ## inverse fft spec
        voltage_Mgm = fftpack.ifft(spec)
        return voltage_Mgm
    def calculateVoltages(self, **kwargs):
        '''
        1) wf fft * response function
        2) ifft to get original wf
        :param kwargs:
        :return:
        '''
        # load info
        if 'voltages_RLC' not in kwargs.keys():
            raise ValueError("Must input RLC voltages to MagnetometerCalculator")
        if 'noises_RLC'    not in kwargs.keys():
            raise ValueError("Must input RLC noises to MagnetometerCalculator")
        voltages_RLC                = np.asarray(kwargs['voltages_RLC'])
        noises_RLC                  = np.asarray(kwargs['noises_RLC'])
        num_samples                 = kwargs.get('num_samples', 1000)
        sample_step                 = kwargs.get('sample_step', 10)  # in ns
        # get frequencies
        freq                        = fftpack.fftfreq(num_samples, sample_step*1e-9)*1e-6 # in MHz
        # signal transfer
        voltages_Mgm                = self._transfer_based_on_voltages(freq, voltages_RLC)
        # noise transfer
        noises_Mgm                  = self._transfer_based_on_voltages(freq, noises_RLC)
        return (voltages_Mgm, noises_Mgm)