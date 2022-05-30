###############################################################
## Module for calculating a particle induced B field or E field
## at different step
## The particle is assumed to pass a long Z+ axis, with starting point at the minimum Z
## of the meshed elements
## by Qing Lin @ 2021-12-02
###############################################################
from tqdm import tqdm
import numpy as np
import yaml
import time
import pickle as pkl
from copy import deepcopy

np.random.seed(int(time.time()))



#########################################
## Field generator
#########################################

class FieldGenerator:
    def __init__(self, **kwargs):
        # read config file
        config = {}
        if 'config_filename' in kwargs.keys():
            config                  = yaml.load(open(kwargs['config_filename'], 'r'), Loader=yaml.FullLoader)
        # check for important inputs
        for key in ['particles']:
            if key not in config.keys():
                raise ValueError(key + " MUST be defined in config file!")
        self._config                = config
        # load all variables
        self._worldBoxSize          = config.get('world_box_size', 400)
        self._cellStep              = config.get('cell_step', 5.0)  # size of each cell
        self._particleStep          = config.get('particle_step', 5.0)  # size of each particle step in flight
        self._particles             = config['particles']  # containing all the configurations, such as direction (rvs range), start point, speed (or energy), charge (or magnetic charge, or magnetic moment) and etc.
        self._saveE                 = config.get('save_E', False)
        self._verbose               = config.get('verbose', False)
        # initiate the finite cell tensors, including Bx, By, Bz, Ex, Ey, Ez
        self._initiateTensors()
        # initiate some physics constants
        self._initiateConstants()
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


    #####################################
    ## Private functions
    #####################################
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


    def _initiateTensors(self):
        '''
        Initiate the tensors of Ex, Ey, Ez, Bx, By, Bz
        :return:
        '''
        # calculate the number of steps on each dimension
        self._numCells          = int(np.floor(self._worldBoxSize / self._cellStep))
        # calculate the bins
        self._cellBins          = np.linspace(-0.5*self._worldBoxSize, 0.5*self._worldBoxSize, self._numCells+1)
        self._cellCenters       = 0.5*(self._cellBins[1:] + self._cellBins[:-1])
        self._cellSteps         = self._cellBins[1:] - self._cellBins[:-1]
        # E & B tensors
        self._E                 = np.zeros((3, self._numCells, self._numCells, self._numCells))
        self._B                 = np.zeros((3, self._numCells, self._numCells, self._numCells))
        # cell center tensor
        x, y, z                 = np.meshgrid(self._cellCenters, self._cellCenters, self._cellCenters, indexing='ij')
        x                       = np.reshape(x, newshape=(1, self._numCells, self._numCells, self._numCells))
        y                       = np.reshape(y, newshape=(1, self._numCells, self._numCells, self._numCells))
        z                       = np.reshape(z, newshape=(1, self._numCells, self._numCells, self._numCells))
        self._cellCenterTensor  = np.concatenate(
            (x,y,z),
            axis=0
        )
        return

    def _initiateOutputDictionary(self):
        '''
        Initiate the dictionary for output
        :return:
        '''
        self._outputDict                        = {}
        # record the config
        self._outputDict['config']              = self._config
        print(self._config)
        # create all the branches
        self._outputDict['part_speed']              = None
        self._outputDict['part_type']               = None
        self._outputDict['part_magnetic_moment']    = None
        self._outputDict['part_electric_charge']    = None
        self._outputDict['part_magnetic_charge']    = None
        self._outputDict['B_tensors']               = []
        if self._saveE:
            self._outputDict['E_tensors']           = []
        return

    def _printEventAndFill(self):
        '''
        Print out the particle information
        And fill the output dictionary about particle
        :param index:
        :return:
        '''
        # print
        print("===>>> particle type = "+str(self._part_type))
        print("===>>> speed = "+str(self._part_speed)+' c')
        print("===>>> electric charge = "+str(self._part_electric_charge)+' e')
        print("===>>> magnetic charge = "+str(self._part_magnetic_charge)+' Dirac magnetic charge')
        print("===>>> magnetic moment = "+str(self._part_magnetic_moment)+' J/T')
        # fill
        self._outputDict['part_type']               = self._part_type
        self._outputDict['part_speed']              = self._part_speed
        self._outputDict['part_electric_charge']    = self._part_electric_charge
        self._outputDict['part_magnetic_charge']    = self._part_magnetic_charge
        self._outputDict['part_magnetic_moment']    = self._part_magnetic_moment
        return

    def _updateTensors(self, initiate):
        '''
        Update the electric & magnetic fields
        Using iteration
        :return:
        '''
        # copy the previous tensors
        self._previous_E                = deepcopy(self._E)
        self._previous_B                = deepcopy(self._B)
        # calculate the current E & B tensor because of particle movement
        self._calcFieldTensorDueToPart()
        if initiate:
            return
        # calculate the induction B
        self._calcInductionB()
        return

    ############################
    ## sub-functions for tensor update
    ############################
    def _calcFieldTensorDueToPart(self):
        '''
        Calculate the first-order field (static) caused by particle itself
        :return:
        '''
        if self._part_type == 0:
            # Monopole
            self._calcFieldTensorDueToMonopole()
        elif self._part_type == 1:
            # charged particle
            self._calcFieldTensorDueToChargedPart()
        elif self._part_type == 2:
            # Magnetic Moment Particle
            self._calcFieldTensorDueToMagneticMomentPart()
        else:
            raise ValueError(
                "Particle type not supported!"
            )
        return

    def _calcFieldTensorDueToMonopole(self):
        '''
        First-order static B field caused by magnetic monopole
        :return:
        '''
        # calculate the vector r
        r                       = deepcopy(self._cellCenterTensor)
        # if self._verbose:
        #     print("before r = "+str(r[2,:,:,:]))
        r[2, :, :, :]           = r[2, :, :, :] - self._partZ  # just change Z
        # if self._verbose:
        #     print("after r = "+str(r[2,:,:,:]))
        # calculate the scalar r
        abs_r                   = np.sqrt(np.sum(r ** 2, axis=0))
        # calculate the B field
        self._B                 = r / abs_r ** 3 * self._reducedPlanckConstant / 2. / self._electronCharge  # in 10^6 kg s^-2 A^-1 = 10^6 T
        # convert to nT
        self._B                 *= 1.e15  # to nT
        # times the magnetic charge
        self._B                 *= self._part_magnetic_charge
        return

    def _calcFieldTensorDueToChargedPart(self):
        '''
        First-order static E field caused by charged particle
        :return:
        '''
        # calculate the vector r
        r                       = deepcopy(self._cellCenterTensor)
        r[2, :, :, :]           = r[2, :, :, :] - self._partZ  # just change Z
        # calculate the scalar r
        abs_r                   = np.sqrt(np.sum(r ** 2, axis=0))
        # calculate the E field
        self._E                 = self._electronCharge / (4. * np.pi * self._dielectricConstant) * r / abs_r ** 3.  # 10^6 V/m
        # convert to SI unit
        self._E                 *= 1.e6  # V/m
        # times the electric charge
        self._E                 *= self._part_electric_charge
        return

    def _calcFieldTensorDueToMagneticMomentPart(self):
        '''
        First-order static B field caused by particles with magnetic moment
        Following formula from Wikipedia:
        https://en.wikipedia.org/wiki/Magnetic_moment
        :return:
        '''
        # calculate the nu_matrix
        mu, _, _, _             = np.meshgrid(
            self._part_magnetic_moment,  # (3)
            self._cellCenters,
            self._cellCenters,
            self._cellCenters,
            indexing='ij'
        )
        # calculate the vector r
        r                       = deepcopy(self._cellCenterTensor)
        r[2, :, :, :]           = r[2, :, :, :] - self._partZ  # just change Z
        # calculate the scalar r
        abs_r                   = np.sum(r ** 2, axis=0)
        # normalized r
        r_hat                   = r / abs_r
        # product of mu and r_hat
        prod_mu_r_hat           = np.sum(mu * r_hat, axis=0)
        prod_mu_r_hat           = np.reshape(prod_mu_r_hat, newshape=(1, self._numCells, self._numCells, self._numCells))
        prod_mu_r_hat           = np.concatenate(
            (prod_mu_r_hat, prod_mu_r_hat, prod_mu_r_hat),
            axis=0
        )
        # calculate the B field
        self._B                 = self._magneticPermiability / (4. * np.pi * abs_r ** 3) * (3. * prod_mu_r_hat * r_hat - mu) * 1e18  # nT
        return

    def _calcInductionB(self):
        '''
        Calculate the induction B field due to E gradients
        Has to assume B_z=0, and assume boundary condition B
        :return:
        '''
        # only if particle type is 1 (charged particle) can do this
        if self._part_type != 1:
            return
        # calculate DeltaE
        DeltaE                  = self._E - self._previous_E  # in V/m
        # calculate delta_t
        Delta_t                 = self._particleStep / (self._speedOfLight * self._part_speed)  # in ns
        # calculate Delta_l
        Delta_l                 = self._cellStep  # in mm
        # calculate the difference matrix
        Delta                   = DeltaE * Delta_l / (self._speedOfLight ** 2 * Delta_t) * 1e-3  # nT
        # calculate the B_x & B_y
        self._iterationProcess(Delta[1, :, :, :], self._numCells - 1, axis=0)
        self._iterationProcess(-Delta[0, :, :, :], self._numCells - 1, axis=1)
        self._B[2, :, :, :]     = 0.
        return

    def _iterationProcess(self, Delta, Index, axis=0):
        '''
        Iteratively calculate the B field matrix element propagation
        :param Delta: Difference matrix of E field (N,N,N)
        :param Index: target index on Z
        :param axis: X or Y
        :return:
        '''
        if Index == 0:
            self._B[axis, :, :, Index] = np.zeros((self._numCells, self._numCells))
            return self._B[axis, :, :, Index]
        elif Index < self._numCells:
            self._B[axis, :, :, Index] = Delta[:, :, Index] + self._iterationProcess(Delta, Index - 1, axis)
            return self._B[axis, :, :, Index]

    ############################
    ############################
    def _save(self):
        '''
        Save each step info to output dictionary
        :return:
        '''
        self._outputDict['B_tensors'].append(deepcopy(self._B))
        if self._saveE:
            self._outputDict['E_tensors'].append(deepcopy(self._E))
        return

    def _outputToFile(self):
        '''
        Output the dictionary to file
        Use pkl here
        :return:
        '''
        pkl.dump(
            self._outputDict,
            open(self._outputFilename,'wb'),
        )
        return



    ####################################
    ## Public function
    ####################################
    def run_sim(self, **kwargs):
        '''
        Run the field simulation and output,
        kwargs include:
        1) output_filename      - output filename, None for not saving
        :param kwargs:
        :return:
        '''
        # load info
        self._outputFilename            = kwargs.get('output_filename', None)
        # initiate the output dictionary
        self._initiateOutputDictionary()
        # get particle info
        self._part_type                 = self._particles.get('part_type', 0)
        self._part_speed                = self._particles.get('speed', 0.001) # unit in light speed
        self._part_electric_charge      = self._particles.get('electric_charge', 1) # unit in +electron charge
        self._part_magnetic_charge      = self._particles.get('magnetic_charge', 1) # unit in +Dirac magnetic charge
        self._part_magnetic_moment      = self._particles.get('magnetic_moment', [0,0,9.284e-24]) # electron magnetic moment along flying direction
        # printout event info
        self._printEventAndFill()
        # loop over particle steps
        for jj in tqdm(range(int(self._worldBoxSize/self._particleStep))):
            self._partZ         = -0.5*self._worldBoxSize + float(jj)*self._particleStep
            # update field tensors
            if jj==0:
                self._updateTensors(initiate=True)
            else:
                self._updateTensors(initiate=False)
            # save
            self._save()
        # output
        self._outputToFile()
        return