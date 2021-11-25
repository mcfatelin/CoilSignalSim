#########################################################
## This module is the master handler for coil simulation
## Basic units used here: mm, ns,  A
## nT and V are derived units,
## 1) Three types of particles are simulated: monopole, charged particle, and particle with magnetic moment
## 2) In the original coordinates, coil is on X-Y plane and centered at (0,0,0), particle direction is defined under this coordinate system.
## 3) When doing finite cell calculation, the coordinate system is transfered to the one with the particle always moving along Z+ direction. Coil centers at (X', 0, 0)
## by Qing Lin
## @ 2021-11-18
########################################################
from tqdm import tqdm
import numpy as np
import yaml
import time
import pickle as pkl
from copy import deepcopy

np.random.seed(int(time.time()))



#######################################
# master handler module
#######################################
class MasterHandler:
    def __init__(self, **kwargs):
        # read config file
        config = {}
        if 'config_filename' in kwargs.keys():
            config                      = yaml.load(open(kwargs['config_filename'], 'r'), Loader=yaml.FullLoader)
        # check for important inputs
        for key in ['particles']:
            if key not in config.keys():
                raise ValueError(key+" MUST be defined in config file!")
        self._config                    = config
        # load all variables
        self._coilDiameter              = config.get('coil_diameter', 100) # in mm
        self._numCoilTurns              = config.get('num_coil_turns', 100)
        self._worldBoxSize              = config.get('world_box_size', self._coilDiameter*4)
        self._cellStep                  = config.get('cell_step', 5.0) # size of each cell
        self._particleStep              = config.get('particle_step', 5.0) # size of each particle step in flight
        self._particles                 = config['particles'] # containing all the configurations, such as direction (rvs range), start point, speed (or energy), charge (or magnetic charge, or magnetic moment) and etc.
        self._verbose                   = config.get('verbose', False)
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

    ######################
    # private functions
    ######################
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
        # create all the branches
        self._outputDict['part_centers']        = [] # original coord frame
        self._outputDict['part_directions']     = [] # original coord frame
        self._outputDict['part_speeds']         = []
        self._outputDict['part_types']          = []
        self._outputDict['part_moment']         = []
        self._outputDict['coil_centers']        = [] # rotated coord frame
        self._outputDict['coil_directions']     = [] # rotated coord frame
        self._outputDict['voltages']            = []
        self._outputDict['part_Zs']             = []
        if self._saveType>0:
            self._outputDict['E_tensors']       = []
            self._outputDict['B_tensors']       = []
        return

    def _initiateEventOutput(self):
        self._outputDict['voltages'].append([])
        self._outputDict['part_Zs'].append([])
        if self._saveType>0:
            self._outputDict['E_tensors'].append([])
            self._outputDict['B_tensors'].append([])
        return

    def _generateParticle(self):
        '''
        Generate particle according to its configuration
        :return:
        '''
        # generate particle start point
        part_coord          = self._generateParticleStartPoint()
        # generate particle direction
        part_direction      = self._generateParticleDirection()
        # generate particle speed
        part_speed          = self._generateParticleSpeed()
        # generate particle magnetic moment
        part_moment         = self._generateParticleMagneticMoment()
        return part_coord, part_direction, part_speed, part_moment

    ##############################
    ## sub-functions for particle generation
    ##############################
    def _generateParticleStartPoint(self):
        generation_type             = self._particles.get('start_point_type', 'point')
        if generation_type=='point':
            part_coord      = np.asarray([
                self._particles.get('start_x', 0),
                self._particles.get('start_y', 0),
                self._particles.get('start_z', -0.5*self._worldBoxSize),
            ])
        elif generation_type=='fixed_x':
            part_coord      = np.asarray([
                self._particles.get('start_x', -0.5*self._worldBoxSize),
                np.random.uniform(
                    self._particles.get('start_y_lower', -0.5*self._worldBoxSize),
                    self._particles.get('start_y_upper', +0.5*self._worldBoxSize),
                ),
                np.random.uniform(
                    self._particles.get('start_z_lower', -0.5*self._worldBoxSize),
                    self._particles.get('start_z_upper', +0.5*self._worldBoxSize),
                ),
            ])
        elif generation_type=='fixed_y':
            part_coord      = np.asarray([
                np.random.uniform(
                    self._particles.get('start_x_lower', -0.5*self._worldBoxSize),
                    self._particles.get('start_x_upper', +0.5*self._worldBoxSize),
                ),
                self._particles.get('start_y', -0.5*self._worldBoxSize),
                np.random.uniform(
                    self._particles.get('start_z_lower', -0.5*self._worldBoxSize),
                    self._particles.get('start_z_upper', +0.5*self._worldBoxSize),
                ),
            ])
        elif generation_type=='fixed_z':
            part_coord      = np.asarray([
                np.random.uniform(
                    self._particles.get('start_x_lower', -0.5*self._worldBoxSize),
                    self._particles.get('start_x_upper', +0.5*self._worldBoxSize),
                ),
                np.random.uniform(
                    self._particles.get('start_y_lower', -0.5*self._worldBoxSize),
                    self._particles.get('start_y_upper', +0.5*self._worldBoxSize),
                ),
                self._particles.get('start_z', -0.5*self._worldBoxSize),
            ])
        elif generation_type=='iso':
            iso_radius          = self._particles.get('start_r', 0.5*self._worldBoxSize)
            rvs_theta           = np.random.uniform(0, np.pi)
            rvs_phi             = np.random.uniform(0, 2.*np.pi)
            part_coord          = np.asarray([
                iso_radius*np.sin(rvs_theta)*np.cos(rvs_phi),
                iso_radius*np.sin(rvs_theta)*np.sin(rvs_phi),
                iso_radius*np.cos(rvs_theta)
            ])
        else:
            raise ValueError("Not supported start point generation type!")
        return part_coord

    def _generateParticleDirection(self):
        generate_type               = self._particles.get('direction_type', 'fixed')
        if generate_type=='fixed':
            part_direction          = np.asarray([
                self._particles.get('direction_x', 0),
                self._particles.get('direction_y', 0),
                self._particles.get('direction_z', 1),
            ])
            part_direction          /= np.sqrt(np.sum(np.power(part_direction, 2.))) # normalization
        elif generate_type=='gaus':
            # with a mean direction and sigma direction
            rvs_theta               = np.random.normal(
                loc                 = self._particles.get('direction_theta', 0),
                scale               = self._particles.get('direction_theta_sigma', 0.1*np.pi),
            )
            rvs_phi                 = np.random.normal(
                loc                 = self._particles.get('direction_phi', 0),
                scale               = self._particles.get('direction_phi_sigma', 0.1*np.pi)
            )
            part_direction          = np.asarray([
                np.sin(rvs_theta)*np.cos(rvs_phi),
                np.sin(rvs_theta)*np.sin(rvs_phi),
                np.cos(rvs_theta)
            ])
        elif generate_type=='iso':
            rvs_theta               = np.random.uniform(0, np.pi)
            rvs_phi                 = np.random.uniform(0, 2.*np.pi)
            part_direction          = np.asarray([
                np.sin(rvs_theta)*np.cos(rvs_phi),
                np.sin(rvs_theta)*np.sin(rvs_phi),
                np.cos(rvs_theta)
            ])
        else:
            raise ValueError("Not supported direction generation type!")
        return part_direction

    def _generateParticleSpeed(self):
        generate_type               = self._particles.get('speed_type', 'fixed')
        if generate_type=='fixed':
            part_speed              = self._particles.get('speed', 1.0) # light of speed
        elif generate_type=='uniform':
            part_speed              = np.random.uniform(
                self._particles.get('speed_lower', 0),
                self._particles.get('speed_upper', 1.0)
            )
        elif generate_type=='gaus':
            part_speed              = np.random.normal(
                loc                 = self._particles.get('speed_mean', 0.5),
                scale               = self._particles.get('speed_sigma', 0.1),
            )
        else:
            raise ValueError("Not supported speed generation type!")
        return part_speed

    def _generateParticleMagneticMoment(self):
        '''
        We always simulated with fixed absolate magnetic moment, meaning a certaint type of particle
        :return:
        '''
        generate_type               = self._particles.get('moment_type', 'fixed')
        if generate_type=='fixed':
            part_moment             = self._particles.get('magnetic_moment', np.asarray([0,0,9.284e-24])) # unit in J/T, default is electron
        elif generate_type=='gaus':
            abs_moment              = self._particles.get('abs_magnetic_moment', 9.284e-24) # J/T
            rvs_theta = np.random.normal(
                loc                 =self._particles.get('moment_direction_theta', 0),
                scale               =self._particles.get('moment_direction_theta_sigma', 0.1 * np.pi),
            )
            rvs_phi = np.random.normal(
                loc                 =self._particles.get('moment_direction_phi', 0),
                scale               =self._particles.get('moment_direction_phi_sigma', 0.1 * np.pi)
            )
            part_moment             = np.asarray([
                np.sin(rvs_theta) * np.cos(rvs_phi),
                np.sin(rvs_theta) * np.sin(rvs_phi),
                np.cos(rvs_theta)
            ])*abs_moment
        elif generate_type=='iso':
            abs_moment = self._particles.get('abs_magnetic_moment', 9.284e-24)  # J/T
            rvs_theta               = np.random.uniform(0, np.pi)
            rvs_phi                 = np.random.uniform(0, 2. * np.pi)
            part_moment             = np.asarray([
                np.sin(rvs_theta) * np.cos(rvs_phi),
                np.sin(rvs_theta) * np.sin(rvs_phi),
                np.cos(rvs_theta)
            ])*abs_moment
        return part_moment

    ##############################
    ##############################


    def _transferCoordinates(self):
        '''
        Transfer the coordinate frame to particle always moving along Z+ axis
        coil centers always on X+ axis
        Update the centers of coil, and
        :return:
        '''
        # initiate the coil coordinates
        self._coil_center               = np.asarray([0,0,0])
        self._coil_direction            = np.asarray([0,0,1.])
        # translate the particle start point to (0,0,0)
        self._coil_center               = self._translate(self._coil_center, -self._part_coord)
        if self._verbose:
            print("==>> After the first translation:")
            self._printCoilInfo()
            print()
        # rotate along Z axis until particle direction projection has no Y component
        if self._part_direction[0]**2+self._part_direction[1]**2<1e-10*self._part_direction[2]**2:
            self._coil_center, self._coil_direction         = self._rotate(
                self._coil_center,
                self._coil_direction,
                angle           = -np.arctan2(self._part_direction[1], self._part_direction[0]),
                axis            = 2,
            )
        if self._verbose:
            print("==>> After rotation along Z:")
            self._printCoilInfo()
            print()
        # rotate along Y until particle direction is along Z
        self._coil_center, self._coil_direction = self._rotate(
            self._coil_center,
            self._coil_direction,
            angle           =-np.arctan2(
                np.sqrt(self._part_direction[0]**2+self._part_direction[1]**2),
                self._part_direction[2],
            ),
            axis            =1,
        )
        if self._verbose:
            print("==>> After rotation along Y:")
            self._printCoilInfo()
            print()
        # rotate along Z until coil center Y=0
        self._coil_center, self._coil_direction = self._rotate(
            self._coil_center,
            self._coil_direction,
            angle           = -np.arctan2(self._coil_center[1], self._coil_center[0]),
            axis            = 2,
        )
        if self._verbose:
            print("==>> After rotation along Z (to make coil center has no Y):")
            self._printCoilInfo()
            print()
        # translate along Z until coil center Z = 0
        self._coil_center               = self._translate(
            self._coil_center,
            np.asarray([0,0,-self._coil_center[2]])
        )
        if self._verbose:
            print("==>> After final translation")
            self._printCoilInfo()
            print()
        return

    ##############################
    ## sub-functions for coordinates transforming
    ##############################
    def _printCoilInfo(self):
        print("==>> coil center = "+str(self._coil_center))
        print("==>> coil direction = "+str(self._coil_direction))

    def _translate(self, x_ori, translation):
        '''
        translate x_ori
        :param x_ori:
        :param translation:
        :return:
        '''
        return x_ori+translation

    def _rotate(self, x1, x2, angle=0, axis=0):
        '''
        rotate along axis to perform a coordinate rotation
        :param x1:
        :param x2:
        :param angle:
        :param axis:
        :return:
        '''
        rotation_matrix     = np.asarray([
            [1., 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), +np.cos(angle)],
        ])
        if axis==1:
            rotation_matrix = np.asarray([
                [+np.cos(angle), 0, +np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, +np.cos(angle)],
            ])
        else:
            rotation_matrix = np.asarray([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), +np.cos(angle), 0],
                [0,0,1.]
            ])
        x1_r                = np.dot(
            rotation_matrix,
            np.reshape(x1, newshape=(3,1))
        )
        x2_r                = np.dot(
            rotation_matrix,
            np.reshape(x2, newshape=(3,1))
        )
        return (
            np.reshape(x1_r, newshape=(-1)),
            np.reshape(x2_r, newshape=(-1)),
        )

    ##############################
    ##############################


    def _printEventAndFill(self, index):
        '''
        Print out the particle information
        And fill the output dictionary about particle and coil info
        :param index:
        :return:
        '''
        self._part_type             = self._particles.get('part_type', 0) # 0 is for monopole, 1 for charged particle, 2 for particle with magnetic moment
        # print
        print("===>>> "+str(index)+" event has been generated")
        print("===>>> start point = "+str(self._part_coord))
        print("===>>> direction = "+str(self._part_direction))
        print("===>>> particle type = "+str(self._part_type))
        print("===>>> speed = "+str(self._part_speed)+' c')
        print("===>>> coil center = "+str(self._coil_center))
        print("===>>> coil direction = "+str(self._coil_direction))
        # fill
        self._outputDict['part_centers'].append(self._part_coord)
        self._outputDict['part_directions'].append(self._part_direction)
        self._outputDict['part_speeds'].append(self._part_speed)
        self._outputDict['part_moment'].append(self._part_mu)
        self._outputDict['part_types'].append(self._part_type)
        self._outputDict['coil_centers'].append(self._coil_center)
        self._outputDict['coil_directions'].append(self._coil_direction)
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
        if self._part_type==0:
            # Monopole
            self._calcFieldTensorDueToMonopole()
        elif self._part_type==1:
            # charged particle
            self._calcFieldTensorDueToChargedPart()
        elif self._part_type==2:
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
        r                   = deepcopy(self._cellCenterTensor)
        # if self._verbose:
        #     print("before r = "+str(r[2,:,:,:]))
        r[2,:,:,:]          = r[2,:,:,:] - self._partZ # just change Z
        # if self._verbose:
        #     print("after r = "+str(r[2,:,:,:]))
        # calculate the scalar r
        abs_r               = np.sqrt(np.sum(r**2, axis=0))
        # calculate the B field
        self._B             = r/abs_r**3 * self._reducedPlanckConstant / 2. / self._electronCharge # in 10^6 kg s^-2 A^-1 = 10^6 T
        # convert to nT
        self._B             *= 1.e15 # to nT
        return

    def _calcFieldTensorDueToChargedPart(self):
        '''
        First-order static E field caused by charged particle
        :return:
        '''
        # calculate the vector r
        r                   = deepcopy(self._cellCenterTensor)
        r[2, :, :, :]       = r[2, :, :, :] - self._partZ  # just change Z
        # calculate the scalar r
        abs_r               = np.sum(r ** 2, axis=0)
        # calculate the E field
        self._E             = self._electronCharge / (4.*np.pi*self._dielectricConstant) * r / abs_r**3. # 10^6 V/m
        # convert to SI unit
        self._E             *= 1.e6 # V/m
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
            self._part_mu, # (3)
            self._cellCenters,
            self._cellCenters,
            self._cellCenters,
            indexing            = 'ij'
        )
        # calculate the vector r
        r                       = deepcopy(self._cellCenterTensor)
        r[2, :, :, :]           = r[2, :, :, :] - self._partZ  # just change Z
        # calculate the scalar r
        abs_r                   = np.sum(r ** 2, axis=0)
        # normalized r
        r_hat                   = r/abs_r
        # product of mu and r_hat
        prod_mu_r_hat           = np.sum(mu*r_hat, axis=0)
        prod_mu_r_hat           = np.reshape(prod_mu_r_hat, newshape=(1, self._numCells, self._numCells, self._numCells))
        prod_mu_r_hat           = np.concatenate(
            (prod_mu_r_hat, prod_mu_r_hat, prod_mu_r_hat),
            axis                = 0
        )
        # calculate the B field
        self._B                 = self._magneticPermiability / (4.*np.pi*abs_r**3) * (3.*prod_mu_r_hat*r_hat - mu)*1e18 # nT
        return

    def _calcInductionB(self):
        '''
        Calculate the induction B field due to E gradients
        Has to assume B_z=0, and assume boundary condition B
        :return:
        '''
        # only if particle type is 1 (charged particle) can do this
        if self._part_type!=1:
            return
        # calculate DeltaE
        DeltaE              = self._E - self._previous_E # in V/m
        # calculate delta_t
        Delta_t             = self._particleStep / (self._speedOfLight * self._part_speed) # in ns
        # calculate Delta_l
        Delta_l             = self._cellStep # in mm
        # calculate the difference matrix
        Delta               = DeltaE * Delta_l / (self._speedOfLight**2 * Delta_t)*1e-3 # nT
        # calculate the B_x & B_
        self._B[0,:,:,-1]   = self._iterationProcess(Delta, self._numCells-1, axis=0)
        self._B[1,:,:,-1]   = self._iterationProcess(Delta, self._numCells-1, axis=1)
        return

    def _iterationProcess(self, Delta, Index, axis=0):
        '''
        Iteratively calculate the B field matrix element propagation
        :param Delta: Difference matrix of E field (N,N,N)
        :param Index: target index on Z
        :param axis: X or Y
        :return:
        '''
        if Index==0:
            return np.zeros((self._numCells, self._numCells))
        elif Index<self._numCells:
            return Delta[axis,:,:,Index] + self._iterationProcess(Delta, Index-1, axis)
        return

    ############################
    ############################

    def _calcInductionVoltage(self):
        '''
        Calculate the induction voltage caused by B field time derivative
        :return:
        '''
        # get the Delta B
        Delta_B                 = self._B - self._previous_B # nT
        # Delta t
        Delta_t                 = self._particleStep / (self._speedOfLight * self._part_speed) # in ns
        # get the index
        xinds, yinds, zinds     = self._getCoilSurfaceIndex()
        # # debug
        # print("xinds = "+str(xinds))
        # print("yinds = "+str(yinds))
        # print("zinds = "+str(zinds))
        # calculate B flux
        BFlux                   = np.sum(Delta_B[:,xinds,yinds,zinds]*self._cellStep**2)
        # convert to right unit
        BFlux                   *= 1.e-15 # V/s
        return -BFlux/Delta_t*1.e9 # in V

    ##############################################
    ## sub-function for induction voltage calculation
    ##############################################
    def _getCoilSurfaceIndex(self):
        # Calculate all the distance
        CoilCenterTensor, _, _, _           = np.meshgrid(
            self._coil_center,
            self._cellCenters,
            self._cellCenters,
            self._cellCenters,
            indexing='ij'
        )
        CoilDirectionTensor, _, _, _        = np.meshgrid(
            self._coil_direction,
            self._cellCenters,
            self._cellCenters,
            self._cellCenters,
            indexing='ij'
        )
        Distance                            = self._cellCenterTensor - CoilCenterTensor
        AbsDistance                         = np.sqrt(
            np.sum((self._cellCenterTensor - CoilCenterTensor) ** 2, axis=0)
        )
        # # debug
        # print("CoilDirectionTensor shape = "+str(CoilDirectionTensor.shape))
        # print("Distance shape = "+str(Distance.shape))
        ProductDirection                    = np.abs(np.sum(
            CoilDirectionTensor*Distance,
            axis=0
        )/AbsDistance)
        # Find the minimum along Z
        zinds                               = np.argmin(ProductDirection,axis=2)
        # # debug
        # print("zinds = "+str(zinds))
        xinds, yinds                        = np.meshgrid(
            np.linspace(0,self._numCells-1, self._numCells).astype(np.int),
            np.linspace(0,self._numCells-1, self._numCells).astype(np.int),
            indexing                        = 'ij'
        )
        # Find the one with distance < r
        inds1, inds2                        = np.where(
            AbsDistance[xinds, yinds, zinds]<0.5*self._coilDiameter
        )
        return xinds[inds1, inds2], yinds[inds1, inds2], zinds[inds1, inds2]
    ##############################################
    ##############################################


    def _save(self):
        '''
        Save each step info to output dictionary
        :return:
        '''
        self._outputDict['voltages'][-1].append(self._voltage)
        self._outputDict['part_Zs'][-1].append(self._partZ)
        if self._saveType>0:
            self._outputDict['E_tensors'][-1].append(deepcopy(self._E))
            self._outputDict['B_tensors'][-1].append(deepcopy(self._B))
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


    ######################
    # public functions
    ######################
    def run_sim(self, **kwargs):
        '''
        Run the simulation and output,
        kwargs include:
        1) num_sim              - number of simulated particle events
        2) output_filename      - output filename, None for not saving
        3) save_type            - type for saving the file, 0 for voltage waveform only, 1 for also field tensor at each particle step
        :param kwargs:
        :return:
        '''
        # load info
        self._numOfSimEvents            = kwargs.get('num_sim', 1)
        self._outputFilename            = kwargs.get('output_filename', None)
        self._saveType                  = kwargs.get('save_type', 0)
        # initiate the output dictionary
        self._initiateOutputDictionary()
        # loop over number of events
        for ii in range(self._numOfSimEvents):
            self._initiateEventOutput()
            # generate particle (direction & start_point)
            self._part_coord, self._part_direction, self._part_speed, self._part_mu        = self._generateParticle()
            # transfer coordinate to particle Z+ frame
            # get the coil centers on X+ axis
            self._transferCoordinates()
            # printout event info
            self._printEventAndFill(ii)
            # loop over particle steps
            for jj in tqdm(range(int(self._worldBoxSize/self._particleStep))):
                self._partZ         = -0.5*self._worldBoxSize + float(jj)*self._particleStep
                # update field tensors
                if jj==0:
                    self._updateTensors(initiate=True)
                else:
                    self._updateTensors(initiate=False)
                # calculate the induction voltage
                self._voltage       = self._calcInductionVoltage()*float(self._numCoilTurns)
                # save
                self._save()
        # output
        self._outputToFile()
        return




