## Module for calculating the Magnetometer response
## do not calculate the noise caused by magnetometer
## by Beige Liu @ 2022-5-14
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
        self.initialconstant()
        return
    def _loadConfig(self,filename):
        ### load the magnetometer parameters
        Dict                    = pkl.load(open(filename,'rb'))
        self.consider_freq      = 4000#Dict['consider_freq']#49694  #in Hz
        self.T2                 = Dict['T2']           #in s
        return
    def initialconstant(self):
        self.ReductionPlankConstant = 6.626/(2*np.pi)*1e-34 # in IS
        self.BhorMagneticMoment     = 9.274*1e-24           # in IS
    def _Response_funtion(self,freq):
        gs = 2.0
        I = 1.5
        gama = gs*self.BhorMagneticMoment/(self.ReductionPlankConstant*(2*I+1))
        ### calculate the complex response value 
        output = self.T2*gama/(-2*1j+2*self.T2*2*np.pi*(self.consider_freq-freq))
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