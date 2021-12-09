###############################################
## code for using existing field tensors
## to calculate the signals induced on coil
## Note it can have modules for LR and magnetometer
###############################################
from tqdm import tqdm
import numpy as np
import yaml
import time
import pickle as pkl
from copy import deepcopy

from modules.InductionCalculator import InductionCalculator
from modules.ArrayManager import ArrayManager
from modules.ParticleGenerator import ParticleGenerator


class SignalCalculator:
    def __init__(self, **kwargs):
        # read config file
        config = {}
        if 'config_filename' in kwargs.keys():
            config = yaml.load(open(kwargs['config_filename'], 'r'), Loader=yaml.FullLoader)
        # check for important inputs
        for key in ['particles', 'field_config', 'array_config']:
            if key not in config.keys():
                raise ValueError(key + " MUST be defined in config file!")
        self._config                = config
        # load all variables
        self._coilDiameter          = config.get('coil_diameter', 100)  # in mm
        self._worldBoxSize          = config.get('world_box_size', 1000)
        self._particles             = config['particles']  # containing all the configurations, such as direction (rvs range), start point, speed (or energy), charge (or magnetic charge, or magnetic moment) and etc.
        self._verbose               = config.get('verbose', False)
        # initiate some physics constants
        self._initiateConstants()
        # load the field tensors
        self._loadFieldTensors(**config['field_config'])
        # load the coil array config
        self._loadArrayConfig(**config['array_config'])
        # load particle generator
        self._loadParticleGenerator(**config['particles'])
        # check if particle generator has the same part_type as in field config
        if self._particleGenerator.getPartType()!=self._inductionCalculator.getPartType():
            raise ValueError(
                "Induction calculator has different particle type than particle generator!"
            )
        return

    def readConfig(self, config_filename):
        '''
        Read config file, only under DEFAULT section
        :param config_filename:
        :return: dict
        '''
        # initiate config parser
        config = configparser.ConfigParser()
        # read config file
        config.read(config_filename)
        # output to dictionary
        Dict = {}
        for key in config['DEFAULT']:
            value = config['DEFAULT'][key]
            if value in ['True', 'TRUE', 'true', 'yes', 'YES', 'Yes']:
                value = True
            elif value in ['False', 'FALSE', 'false', 'no', 'NO', 'No']:
                value = False
            else:
                try:
                    value = eval(value)
                except:
                    pass
            Dict[key] = value
        return Dict

    ###############################################
    ## Loader
    ###############################################
    def _initiateConstants(self):
        '''
        initiate the physics constants
        :return:
        '''
        self._speedOfLight          = 2.998e2 # mm/ns
        self._dielectricConstant    = 8.854e-12 # F/m = A^2 s^4 m^-3 kg^-1
        self._magneticPermiability  = 1.2566e-6 # N/A^2
        self._electronCharge        = 1.602e-19 # C
        self._reducedPlanckConstant = 1.0546e-34 # J s
        return


    def _loadParticleGenerator(self, **kwargs):
        '''
        Load particle generator
        :param kwargs:
        :return:
        '''
        self._particleGenerator                 = ParticleGenerator(**kwargs)
        return

    def _loadFieldTensors(self, **kwargs):
        '''
        Load the config for field tensor, more specifically magnetic field tensor
        :param kwargs:
        :return:
        '''
        self._inductionCalculator               = InductionCalculator(**kwargs)
        return

    def _loadArrayConfig(self, **kwargs):
        '''
        Load the config for arrays
        :param kwargs:
        :return:
        '''
        self._arrayManager                      = ArrayManager(**kwargs)
        return


    ###############################################
    ## Private functions
    ###############################################
    def _initiateOutputDictionary(self):
        '''
        Initiate the dictionary for output
        :return:
        '''
        self._outputDict = {}
        # record the config
        self._outputDict['config']              = self._config
        # create all the branches
        self._outputDict['part_centers']            = []  # original coord frame
        self._outputDict['part_directions']         = []  # original coord frame
        self._outputDict['part_speeds']             = []
        self._outputDict['part_types']              = self._particleGenerator.getPartType()
        self._outputDict['part_magnetic_moments']   = []
        self._outputDict['part_electric_charges']   = []
        self._outputDict['part_magnetic_charges']   = []
        self._outputDict['coil_ids']                = [] # ids of hit coils
        self._outputDict['hit_times']               = [] # hit times of each coil
        self._outputDict['voltages']                = []
        # self._outputDict['voltages_LR']             = [] # to be implemented
        # self._outputDict['voltage_magnetometer']    = [] # to be implemented
        return

    def _generateParticle(self):
        '''
        Generate particle according to its configuration
        :return:
        '''
        # generate particle start point
        self._generateParticleStartPoint()
        # generate particle direction
        self._generateParticleDirection()
        # generate particle speed
        self._generateParticleSpeed()
        # generate particle electric charge
        self._generateParticleElectricCharge()
        # generate particle magnetic charge
        self._generateParticleMagneticCharge()
        # generate particle magnetic moment
        self._generateParticleMagneticMoment()
        return

    ##########################################
    ## sub functions for particle generation
    ##########################################
    def _generateParticleStartPoint(self):
        '''
        Generate particle start point
        :return:
        '''
        self._part_center           = self._particleGenerator.generateStartPoint()
        return

    def _generateParticleDirection(self):
        '''
        Generate particle direction
        :return:
        '''
        self._part_direction        = self._particleGenerator.generateDirection()
        return

    def _generateParticleSpeed(self):
        '''
        Generate particle speed
        :return:
        '''
        self._part_speed            = self._particleGenerator.generateSpeed()
        return

    def _generateParticleElectricCharge(self):
        '''
        Generate particle electric charge
        :return:
        '''
        self._part_electric_charge  = self._particleGenerator.generateElectricCharge()
        return

    def _generateParticleMagneticCharge(self):
        '''
        Generate particle magnetic charge
        :return:
        '''
        self._part_magnetic_charge  = self._particleGenerator.generateMagneticCharge()
        return

    def _generateParticleMagneticMoment(self):
        '''
        Generate particle magnetic moment
        :return:
        '''
        self._part_magnetic_moment  = self._particleGenerator.generateMagneticMoment()
        return


    ##########################################
    ##########################################


    def _printEvent(self, index):
        '''
        Print out the particle information
        And fill the output dictionary about particle and coil info
        :param index:
        :return:
        '''
        # print
        print("===>>> " + str(index) + " event has been generated")
        print("===>>> start point = " + str(self._part_center))
        print("===>>> direction = " + str(self._part_direction))
        print("===>>> particle type = " + str(self._particleGenerator.getPartType()))
        print("===>>> speed = " + str(self._part_speed) + ' c')
        print("===>>> particle electric charge = " + str(self._part_electric_charge) + ' electron charge')
        print("===>>> particle magnetic charge = " + str(self._part_magnetic_charge) + ' Dirac magnetic charge')
        print("===>>> particle magnetic moment = " + str(self._part_magnetic_moment) + ' Bohr moment')
        return

    def _calcInductionVoltages(self):
        '''
        Calculate the induction voltages
        :return:
        '''
        # array manager to calculate the "hit" coils
        self._arrayManager.calcHitCoils(
            part_start_point            = self._part_center,
            part_direction              = self._part_direction,
            part_speed                  = self._part_speed*self._speedOfLight, # mm/ns
            tolerance                   = self._inductionCalculator.getBoxSize()/2.
        )
        # loop over each hit
        self._voltages                  = []
        self._hit_times                 = []
        self._coil_ids                  = []
        for hit_info in self._arrayManager.getHits():
            # load info
            start_point                 = hit_info['hit_start_point']
            self._hit_times.append(hit_info['hit_time'])
            self._coil_ids.append(hit_info['coil_id'])
            # calculate voltage
            self._voltages.append(
                self._inductionCalculator.calcInductionVoltage(
                    start_point         = start_point,
                    direction           = self._part_direction,
                    coil_center         = hit_info['coil_center'],
                    coil_direction      = hit_info['coil_direction'],
                    coil_diameter       = self._coilDiameter,
                    part_speed          = self._part_speed,
                    part_electric_charge= self._part_electric_charge,
                    part_magnetic_charge= self._part_magnetic_charge,
                    part_magnetic_moment= self._part_magnetic_moment
                )
            )
        return

    def _save(self):
        self._outputDict['part_centers'].append(self._part_center)
        self._outputDict['part_directions'].append(self._part_direction)
        self._outputDict['part_speeds'].append(self._part_speed)
        self._outputDict['part_electric_charges'].append(self._part_electric_charge)
        self._outputDict['part_magnetic_charges'].append(self._part_magnetic_charge)
        self._outputDict['part_magnetic_moments'].append(self._part_magnetic_moment)
        self._outputDict['coil_ids'].append(deepcopy(self._coil_ids))
        self._outputDict['hit_times'].append(deepcopy(self._hit_times))
        self._outputDict['voltages'].append(deepcopy(self._voltages))
        return


    def _outputToFile(self):
        '''
        Output the dictionary to file
        Use pkl here
        :return:
        '''
        pkl.dump(
            self._outputDict,
            open(self._outputFilename, 'wb'),
        )
        return



    ###############################################
    ## Public function
    ###############################################

    def run_sim(self, **kwargs):
        '''
        Run the calculation of induced signal
        kwargs include:
        1) output_filename      - output filename, None for not saving
        2) num_sim              - Number of simulated events
        :param kwargs:
        :return:
        '''
        # load info
        self._numOfSimEvents                = kwargs.get('num_sim', 1)
        self._outputFilename                = kwargs.get('output_filename', None)
        # initiate the output dictionary
        self._initiateOutputDictionary()
        # loop over number of events
        for ii in tqdm(range(self._numOfSimEvents)):
            # generate particle (direction & start_point)
            self._generateParticle()
            # printout event info
            if self._verbose:
                self._printEvent(ii)
            # calculate induction voltages on all "hit" coils
            self._calcInductionVoltages()
            # save
            self._save()
        # output
        self._outputToFile()
        return

