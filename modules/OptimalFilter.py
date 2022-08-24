##Module to calculate optimal filter of the signal 
##by Beige Liu @20220704
import numpy as np
from  scipy import fftpack
import pickle as pkl
from scipy.interpolate import interp1d
from copy import deepcopy
class OptimalFilter():
    def __init__(self,**kwargs):
        if 'filename' in kwargs.keys():
            self._loadConfig(kwargs['filename'])
        else:
            print('OptimalFilter config filename must be given')
        return
    def _loadConfig(self, filename):
        '''
        Load config file, dumping the amplitude and phase vs freq
        :param filename:
        :return:
        '''
        # open file
        Dict               = pkl.load(open(filename, 'rb'))
        # Dump info
        self._signalInterpolates            = {}
        self._signalInterpolates['amp']     = interp1d(
                    Dict["respond"]['f'],
                    Dict["respond"]['amp'],
                    bounds_error        = False,
                    kind                = 'quadratic',
                    fill_value          = (Dict["respond"]['amp'][0], Dict["respond"]['amp'][-1]),
                )
        self._signalInterpolates['phase_x'] = interp1d(
                    Dict["respond"]['f'],
                    Dict["respond"]['phase_x'],
                    bounds_error        = False,
                    fill_value          = (Dict["respond"]['phase_x'][0], Dict["respond"]['phase_x'][-1]),
                )
        self._signalInterpolates['phase_y'] = interp1d(
                    Dict["respond"]['f'],
                    Dict["respond"]['phase_y'],
                    bounds_error        = False,
                    fill_value          = (Dict["respond"]['phase_y'][0], Dict["respond"]['phase_y'][-1]),
                )
        return
    def _conjugate(self, freq, spec):
        '''
        conjugate the power spec
        if positive and negative freq part has different value, stick on positive
        :param freq:
        :param values:
        :return:
        '''
        # generate a tracing index table
        for i in range(len(freq)):
            if freq[i]<0:
                spec[i] = np.conjugate(spec[i])
        return spec
    def _transfer_based_on_voltages(self, freq, voltages,respond_dict):
        '''
        fft->response->ifft
        :param voltages:
        :return:
        '''
        # first fft voltage to spec
        spec                = fftpack.fft(voltages)
        #print(spec[100:110])
        # get the response
        amp                 = respond_dict['amp'](np.abs(freq))
        phase_x             = respond_dict['phase_x'](np.abs(freq))
        phase_y             = respond_dict['phase_y'](np.abs(freq))
        phase               = phase_x + 1j*phase_y
        # multiply the spec with response function
        phase               = self._conjugate(freq,phase)
        spec                = np.multiply(
            spec,
            amp*phase
        )
        # inverse fft
        voltages_Op        = fftpack.ifft(spec)
        return voltages_Op
    def calculateVoltages(self, **kwargs):
        '''
        1) wf fft * response function
        2) ifft to get original wf
        :param kwargs:
        :return:
        '''
        # load info
        if 'voltages_magnetometer' not in kwargs.keys():
            raise ValueError("Must input voltages_magnetometer to Optimalfilter Calculator")
        voltages                    = np.asarray(kwargs['voltages_magnetometer'])
        noises                      = np.asarray(kwargs['noises_magnetometer'])
        num_samples                 = kwargs.get('num_samples', 1000)
        sample_step                 = kwargs.get('sample_step', 100) # in ns
        # get frequencies
        freq                        = fftpack.fftfreq(num_samples, sample_step*1e-9) # in MHz
        # signal transfer
        voltages_Op                = self._transfer_based_on_voltages(freq, voltages,self._signalInterpolates )
        # noise transfer
        noises_Op                  = self._transfer_based_on_voltages(freq, noises,self._signalInterpolates )
        return voltages_Op,noises_Op
