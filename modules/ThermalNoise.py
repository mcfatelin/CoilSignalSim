#################################################
## This module handles adding the Johnson (thermal) noise
## by Qing Lin @ 2021-12-22
##################################################
import numpy as np




class ThermalNoiseGenerator:
    def __init__(self, **kwargs):
        self._resistance                = kwargs.get('R', 1000) # Om
        self._temperature               = kwargs.get('T', 300) # K
        self._sampling_rate             = kwargs.get('sampling_rate', 10000000) # Hz
        self._bandwidth                 = 0.5*self._sampling_rate # bandwidth is half of the sampling rate
        self._initiateConstants()
        return


    ##################################
    ## Private functions
    ###################################
    def _initiateConstants(self):
        '''
        initiate the physics constants
        :return:
        '''
        self._speedOfLight                  = 2.998e2  # mm/ns
        self._dielectricConstant            = 8.854e-12  # F/m = A^2 s^4 m^-3 kg^-1
        self._magneticPermiability          = 1.2566e-6  # N/A^2
        self._electronCharge                = 1.602e-19  # C
        self._reducedPlanckConstant         = 1.0546e-34  # J s
        self._boltzmannConstant             = 1.380649e-23 # J K^-1
        return



    ###################################
    ## Public functions
    ###################################
    def generateNoise(self, **kwargs):
        '''
        Generate thermal noise
        can overwrite the sampling rate
        :param kwargs:
        :return:
        '''
        # load info
        self._size                  = int(kwargs.get('size', 100))
        self._sampling_rate         = kwargs.get('sampling_rate', self._sampling_rate)
        # calculate the sigma of voltage
        voltage_sigma               = np.sqrt(
            4.*self._boltzmannConstant*self._temperature*self._resistance*self._bandwidth
        ) # in V
        # sample the white noise
        if voltage_sigma>0:
            voltages                    = np.random.normal(
                loc                     = 0.,
                scale                   = voltage_sigma,
                size                    = self._size
            )
        else:
            voltages                    = np.zeros(shape=self._size)
        return voltages


