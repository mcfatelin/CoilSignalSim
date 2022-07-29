#################################################################################
## Module for calculating the RLC circuit response
## Effective serial and parallel branches are okay, but tree-level or loop-level are not supported
## Data structure use "tree" structure to store effective L, R, C, and I, U
## by Qing Lin @ 2021-12-23
#############################
## Updated @ 2022-04-01
## Now we took from external circuit simulation:
## 1) Amplitude vs freq
## 2) Phase vs freq
## as the response function
## Apply this upon signal and noise frequency dist. and inverse fft to get the shaped signal and noise wf
## Yes noise is also generated here.
#################################################################################
import numpy as np
import pickle as pkl
import scipy
from scipy.interpolate import interp1d
from scipy import fftpack
from copy import deepcopy



class CircuitCalculator:
    # place-holder
    def __init__(self, **kwargs):
        self._signalInterpolates            = {}
        self._noiseInterpolates             = []
        if 'filename' not in kwargs.keys():
            raise ValueError(
                "Must give circuit config filename!"
            )
        self._loadConfig(kwargs['filename'])
        # some constants
        self._kB                            = 1.380649e-23 # J/K
        return

    ########################################
    ## Private functions
    ########################################
    def _loadConfig(self, filename):
        '''
        Load config file, dumping the amplitude and phase vs freq
        :param filename:
        :return:
        '''
        # open file
        Dict               = pkl.load(open(filename, 'rb'))
        # Dump info
        for key in Dict.keys():
            if key=='signal':
                self._signalInterpolates            = {}
                self._signalInterpolates['amp']     = interp1d(
                    Dict[key]['f'],
                    Dict[key]['amp'],
                    bounds_error        = False,
                    fill_value          = (Dict[key]['amp'][0], Dict[key]['amp'][-1]),
                )
                self._signalInterpolates['phase_x'] = interp1d(
                    Dict[key]['f'],
                    Dict[key]['phase_x'],
                    bounds_error        = False,
                    fill_value          = (Dict[key]['phase_x'][0], Dict[key]['phase_x'][-1]),
                )
                self._signalInterpolates['phase_y'] = interp1d(
                    Dict[key]['f'],
                    Dict[key]['phase_y'],
                    bounds_error        = False,
                    fill_value          = (Dict[key]['phase_y'][0], Dict[key]['phase_y'][-1]),
                )
            else:
                for InputDict in Dict[key]:
                    InterpolatesDict                    = {}
                    InterpolatesDict['Rdc']             = InputDict['Rdc'] # in Om
                    InterpolatesDict['Rslope']          = interp1d(
                        InputDict['f'],
                        InputDict['Rslope'],
                        bounds_error        = False,
                        fill_value          = (InputDict['Rslope'][0], InputDict['Rslope'][-1]))
                    InterpolatesDict['Temp']            = InputDict['Temp'] # in K
                    InterpolatesDict['amp']             = interp1d(
                        InputDict['f'],
                        InputDict['amp'],
                        bounds_error        = False,
                        fill_value          = (InputDict['amp'][0], InputDict['amp'][-1]),
                    )
                    InterpolatesDict['phase_x']         = interp1d(
                        InputDict['f'],
                        InputDict['phase_x'],
                        bounds_error        = False,
                        fill_value          = (InputDict['phase_x'][0], InputDict['phase_x'][-1])
                    )
                    InterpolatesDict['phase_y']         = interp1d(
                        InputDict['f'],
                        InputDict['phase_y'],
                        bounds_error        = False,
                        fill_value          = (InputDict['phase_y'][0], InputDict['phase_y'][-1])
                    )
                    self._noiseInterpolates.append(deepcopy(InterpolatesDict))
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

    def _generateNoiseFrequencySpec(self, freq, Rslope_dict,**kwargs):
        '''
        Generate Johnson thermal noise
        Note: we need to scale the spectrum because of fft convention
        :param freq:
        :param kwargs:
        :return:
        '''
        # load info
        Rdc                 = kwargs.get('Rdc', 1) # in Om
        Rslope              = Rslope_dict(np.abs(freq))
        Temp                = kwargs.get('Temp', 300) # in K
        num_samples         = kwargs.get('num_samples', 1000)
        sample_step         = kwargs.get('sample_step', 10) # in ns
        sample_rate         = 1./sample_step*1e9 # in Hz
        # Generate random phase
        phase               = np.exp(1j*np.random.uniform(0, 2.*np.pi, size=freq.shape[0]))
        phase               = self._conjugate(freq, phase)
        # Get the power spec
        spec                = np.multiply(
            np.sqrt(4.*self._kB*Temp*Rslope*Rdc*sample_rate*num_samples),
            phase
        )
        return spec
    def _transfer_based_on_voltages(self, freq, voltages, respond_dict):
        '''
        fft->response->ifft
        :param voltages:
        :return:
        '''
        # first fft voltage to spec
        spec                = fftpack.fft(voltages)
        # get the response
        amp                 = respond_dict['amp'](np.abs(freq))
        phase_x             = respond_dict['phase_x'](np.abs(freq))
        phase_y             = respond_dict['phase_y'](np.abs(freq))
        # conjugate phase_y
        phase               = phase_x + 1j*phase_y
        phase               = self._conjugate(freq, phase)
        # multiply the spec with response function
        spec                = np.multiply(
            spec,
            amp*phase
        )
        # inverse fft
        voltages_RLC        = fftpack.ifft(spec)
        return voltages_RLC

    def _transfer_based_on_spec(self, freq, spec, respond_dict):
        '''
        response->ifft
        :param spec:
        :return:
        '''
        # get the response
        amp                 = respond_dict['amp'](np.abs(freq))
        phase_x             = respond_dict['phase_x'](np.abs(freq))
        phase_y             = respond_dict['phase_y'](np.abs(freq))
        # conjugate phase_y
        phase               = phase_x + 1j*phase_y
        phase               = self._conjugate(freq, phase)
        # multiply the spec with response function
        spec = np.multiply(
            spec,
            amp*phase
        )
        # inverse fft
        voltages_RLC = fftpack.ifft(spec)
        return voltages_RLC

    #######################################
    ## Public functions
    #######################################
    def calculateVoltages(self, **kwargs):
        '''
        1) wf fft * response function
        2) ifft to get original wf
        :param kwargs:
        :return:
        '''
        # load info
        if 'voltages' not in kwargs.keys():
            raise ValueError("Must input voltages to CircuitCalculator")
        voltages                    = np.asarray(kwargs['voltages'])
        num_samples                 = kwargs.get('num_samples', 1000)
        sample_step                 = kwargs.get('sample_step', 100) # in ns
        # get frequencies
        freq                        = fftpack.fftfreq(num_samples, sample_step*1e-9)*1e-6 # in MHz
        # signal transfer
        voltages_RLC                = self._transfer_based_on_voltages(freq, voltages, self._signalInterpolates)
        # noise transfer
        noises_RLC                  = np.zeros(voltages.shape)
        for InterpolatorDict in self._noiseInterpolates:
            noise_freq_spec         = self._generateNoiseFrequencySpec(
                freq,
                Rdc                 = InterpolatorDict['Rdc'],
                Rslope_dict         = InterpolatorDict['Rslope'],
                Temp                = InterpolatorDict['Temp'],
                num_samples         = num_samples,
                sample_step         = sample_step,
            )
            # print("noise_freq_spec = "+str(noise_freq_spec))
            # for f, spec in zip(freq, noise_freq_spec):
            #     print("f: "+str(f)+"; noise="+str(spec))
            this_noise_RLC          = self._transfer_based_on_spec(freq, noise_freq_spec, InterpolatorDict)
            # print("this_noise_RLC = "+str(this_noise_RLC))
            noises_RLC              = noises_RLC + this_noise_RLC
        return (voltages_RLC, noises_RLC)



