###########################################################
## Particle generator for sampling the particle's start point, direction, speed, etc.
## by Qing Lin @ 2021-12-07
############################################################
import numpy as np
import pickle as pkl
from scipy.interpolate import interp1d


class GeneralGenerator:
    def __init__(self, input):
        if type(input)==int or type(input)==float:
            # constant
            self._type              = 0
            self._value             = float(input)
        elif type(input)==list and len(input)==2:
            # uniform type
            self._type              = 1
            self._lower             = float(input[0])
            self._upper             = float(input[1])
        elif type(input)==dict:
            # gaussian type
            self._type              = 2
            self._mean              = input['mean']
            self._std               = input['std']
        elif type(input)==str:
            # customized type
            self._type              = 3
            self._loadCustomizedInput(input)
        else:
            raise ValueError(str(input)+" not supported!")
        return

    def _loadCustomizedInput(self, filename):
        '''
        Load a file containing customized sampling pdf (pickle file)
        :param filename:
        :return:
        '''
        # open file
        Dict                = pkl.load(open(filename, 'rb'))
        # make cdf
        x                   = np.asarray(Dict['x'])
        cdf                 = np.cumsum(Dict['y']) / np.sum(Dict['y'])
        # make interpolator
        self._interpolator  = interp1d(
            cdf,
            x,
            bounds_error    = False,
        )
        return

    def sample(self):
        '''
        Sample according to pre-stored inputs
        :return:
        '''
        if self._type==0:
            return self._value
        elif self._type==1:
            return np.random.uniform(
                self._lower,
                self._upper
            )
        elif self._type==2:
            return np.random.normal(
                loc         = self._mean,
                scale       = self._std,
            )
        elif self._type==3:
            rvs             = np.random.uniform(0,1)
            return self._interpolator(rvs)


class ParticleGenerator:
    def __init__(self, **kwargs):
        self._loadConfig(**kwargs)
        return


    #########################################
    ## Private functions
    #########################################
    def _loadConfig(self, **kwargs):
        '''
        Load the config
        :param kwargs:
        :return:
        '''
        # initiate all variables
        self._startXGenerator                       = None
        self._startYGenerator                       = None
        self._startZGenerator                       = None
        self._startRGenerator                       = None
        self._startThetaGenerator                   = None
        self._startPhiGenerator                     = None
        self._directionXGenerator                   = None
        self._directionYGenerator                   = None
        self._directionZGenerator                   = None
        self._directionThetaGenerator               = None
        self._directionPhiGenerator                 = None
        self._speedGenerator                        = None
        self._electricChargeGenerator               = None
        self._magneticChargeGenerator               = None
        self._absMagneticMomentGenerator            = None
        self._momentDirectionXGenerator             = None
        self._momentDirectionYGenerator             = None
        self._momentDirectionZGenerator             = None
        self._momentDirectionThetaGenerator         = None
        self._momentDirectionPhiGenerator           = None
        # start point
        if 'start_point_x' in kwargs.keys():
            self._startXGenerator       = GeneralGenerator(kwargs.get('start_point_x', 0))
            self._startYGenerator       = GeneralGenerator(kwargs.get('start_point_y', 0))
            self._startZGenerator       = GeneralGenerator(kwargs.get('start_point_z', 0))
        else:
            self._startRGenerator       = GeneralGenerator(kwargs.get('start_point_r', 400.))
            self._startThetaGenerator   = GeneralGenerator(kwargs.get('start_point_theta', 0))
            self._startPhiGenerator     = GeneralGenerator(kwargs.get('start_point_phi', 0))
        # direction
        if 'direction_x' in kwargs.keys():
            self._directionXGenerator   = GeneralGenerator(kwargs.get('direction_x', 0))
            self._directionYGenerator   = GeneralGenerator(kwargs.get('direction_y', 0))
            self._directionZGenerator   = GeneralGenerator(kwargs.get('direction_z', 1))
        else:
            self._directionThetaGenerator       = GeneralGenerator(kwargs.get('direction_theta', 0))
            self._directionPhiGenerator         = GeneralGenerator(kwargs.get('direction_phi', 0))
        # speed
        self._speedGenerator                    = GeneralGenerator(kwargs.get('speed', 0.001))
        # electric charge
        self._electricChargeGenerator           = GeneralGenerator(kwargs.get('electric_charge', 1))
        # magnetic charge
        self._magneticChargeGenerator           = GeneralGenerator(kwargs.get('magnetic_charge', 1))
        # magnetic moment
        self._absMagneticMomentGenerator        = GeneralGenerator(kwargs.get('abs_magnetic_moment', 9.284e-24))
        if 'moment_direction_x' in kwargs.keys():
            self._momentDirectionXGenerator     = GeneralGenerator(kwargs.get('moment_direction_x', 0))
            self._momentDirectionYGenerator     = GeneralGenerator(kwargs.get('moment_direction_y', 0))
            self._momentDirectionZGenerator     = GeneralGenerator(kwargs.get('moment_direction_z', 1))
        else:
            self._momentDirectionThetaGenerator = GeneralGenerator(kwargs.get('moment_direction_theta', 0))
            self._momentDirectionPhiGenerator   = GeneralGenerator(kwargs.get('moment_direction_phi', 0))
        # part_type
        if 'part_type' in kwargs.keys():
            self._partType                          = kwargs['part_type']
        else:
            raise ValueError(
                'Particle type must be defined!'
            )
        return



    #########################################
    ## Public functions
    #########################################
    def getPartType(self):
        return self._partType

    def generateStartPoint(self):
        '''
        Generate particle start point
        :return:
        '''
        if self._startXGenerator is not None:
            x               = self._startXGenerator.sample()
            y               = self._startYGenerator.sample()
            z               = self._startZGenerator.sample()
        else:
            r               = self._startRGenerator.sample()
            theta           = self._startThetaGenerator.sample()
            phi             = self._startPhiGenerator.sample()
            x               = r*np.sin(theta)*np.cos(phi)
            y               = r*np.sin(theta)*np.sin(phi)
            z               = r*np.cos(theta)
        return np.asarray([x,y,z])

    def generateDirection(self):
        '''
        Generate direction
        :return:
        '''
        if self._directionXGenerator is not None:
            px              = self._directionXGenerator.sample()
            py              = self._directionYGenerator.sample()
            pz              = self._directionZGenerator.sample()
            p               = np.sqrt(px**2+py**2+pz**2)
            px              /= p
            py              /= p
            pz              /= p
        else:
            ptheta          = self._directionThetaGenerator.sample()
            pphi            = self._directionPhiGenerator.sample()
            px              = np.sin(ptheta)*np.cos(pphi)
            py              = np.sin(ptheta)*np.sin(pphi)
            pz              = np.cos(ptheta)
        return np.asarray([px,py,pz])

    def generateSpeed(self):
        '''
        Generate speed
        :return:
        '''
        return self._speedGenerator.sample()

    def generateElectricCharge(self):
        '''
        Generate electric charge
        :return:
        '''
        return self._electricChargeGenerator.sample()

    def generateMagneticCharge(self):
        '''
        Generate magnetic charge
        :return:
        '''
        return self._magneticChargeGenerator.sample()

    def generateMagneticMoment(self):
        '''
        Generate magnetic moment
        :return:
        '''
        abs_magnetic_moment             = self._absMagneticMomentGenerator.sample()
        if self._momentDirectionXGenerator is not None:
            px                          = self._momentDirectionXGenerator.sample()
            py                          = self._momentDirectionYGenerator.sample()
            pz                          = self._momentDirectionZGenerator.sample()
            p                           = np.sqrt(px**2+py**2+pz**2)
            px                          /= p
            py                          /= p
            pz                          /= p
        else:
            ptheta                      = self._momentDirectionThetaGenerator.sample()
            pphi                        = self._momentDirectionPhiGenerator.sample()
            px                          = np.sin(ptheta)*np.cos(pphi)
            py                          = np.sin(ptheta)*np.sin(pphi)
            pz                          = np.cos(ptheta)
        return np.asarray([px*abs_magnetic_moment, py*abs_magnetic_moment, pz*abs_magnetic_moment])