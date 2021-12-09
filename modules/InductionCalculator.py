#########################################################
## Module for calculating the amount of voltages induced by Magnetic field derivative
## by Qing Lin @ 2021-12-07
##########################################################
import numpy as np
import pickle as pkl
from copy import deepcopy




class InductionCalculator:
    def __init__(self, **kwargs):
        if kwargs.get('multiple', False):
            self._calculator            = MultipleInductionCalculator(**kwargs)
        else:
            self._calculator            = SingleInductionCalculator(**kwargs)
        return

    ##########################
    ## Public
    ##########################
    def getPartType(self):
        return self._calculator.getPartType()

    def getBoxSize(self):
        return self._calculator.getBoxSize()

    def calcInductionVoltage(self, **kwargs):
        return self._calculator.calcInductionVoltage(**kwargs)



#################################################
## Single Induction calculator
##################################################

class SingleInductionCalculator:
    def __init__(self, **kwargs):
        self._loadConfig(**kwargs)
        self._initiateConstants()
        return


    ###################################################
    ## Private functions
    ###################################################
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


    def _loadConfig(self, **kwargs):
        '''
        Load info from a file
        :param kwargs:
        :return:
        '''
        # load info
        self._numCoilTurns              = kwargs.get('num_coil_turns', 100)
        # load file
        Dict                            = pkl.load(open(kwargs['filename'], 'rb'))
        # dump info
        self._worldBoxSize              = Dict['config']['world_box_size']
        self._cellStep                  = Dict['config']['cell_step']
        self._particleStep              = Dict['config']['particle_step']
        self._partType                  = Dict['part_type']
        self._partSpeed                 = Dict['part_speed']
        self._partElectricCharge        = Dict['part_electric_charge']
        self._partMagneticCharge        = Dict['part_magnetic_charge']
        self._partMagneticMoment        = np.asarray(Dict['part_magnetic_moment'])
        self._B                         = np.asarray(Dict['B_tensors']) #
        # make secondary derivative info
        self._numCells                  = int(np.floor(self._worldBoxSize/ self._cellStep))
        self._cellBins                  = np.linspace(-0.5*self._worldBoxSize, 0.5*self._worldBoxSize, self._numCells+1)
        self._cellCenters               = 0.5*(self._cellBins[1:]+self._cellBins[:-1])
        self._cellSteps                 = self._cellBins[1:] - self._cellBins[:-1]
        # cell center tensor
        x, y, z                         = np.meshgrid(self._cellCenters, self._cellCenters, self._cellCenters, indexing='ij')
        x                               = np.reshape(x, newshape=(1, self._numCells, self._numCells, self._numCells))
        y                               = np.reshape(y, newshape=(1, self._numCells, self._numCells, self._numCells))
        z                               = np.reshape(z, newshape=(1, self._numCells, self._numCells, self._numCells))
        self._cellCenterTensor          = np.concatenate(
            (x, y, z),
            axis=0
        )
        return

    def _transferCoordinates(self, **kwargs):
        '''
        Transfer the coordinate to the one
        so that the particle moves along Z+
        and coil center is on X+
        :param kwargs:
        :return:
        '''
        # load info
        part_start_point                = kwargs['part_start_point'] # should be the closest point of particle with respect to coil center
        part_direction                  = kwargs['part_direction']
        coil_center                     = kwargs['coil_center']
        coil_direction                  = kwargs['coil_direction']
        # translate the particle start point to (0,0,0)
        coil_center                     = self._translate(coil_center, -part_start_point)
        # rotate along Z axis until particle direction projection has no Y component
        if part_direction[0]**2 + part_direction[1]**2 < 1e-10 * part_direction[2]**2:
            coil_center, coil_direction = self._rotate(
                coil_center,
                coil_direction,
                angle                   = -np.arctan2(part_direction[1], part_direction[0]),
                axis                    = 2
            )
        # rotate along Y until particle direction along Z
        coil_center, coil_direction     = self._rotate(
            coil_center,
            coil_direction,
            angle                       = np.arctan2(
                np.sqrt(part_direction[0]**2+part_direction[1]**2),
                part_direction[2]
            ),
            axis                        = 1
        )
        # rotate along Z until coil center Y=0
        coil_center, coil_direction     = self._rotate(
            coil_center,
            coil_direction,
            angle                       = -np.arctan2(coil_center[1], coil_center[0]),
            axis                        = 2.,
        )
        # translate along Z until coil center Z=0
        # We hope we don't need to do this
        return coil_center, coil_direction
    ################################
    ## sub-functions
    ################################
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

    ################################
    ################################


    def _getCoilSurfaceIndex(self, **kwargs):
        # load inf
        coil_center                     = kwargs['coil_center']
        coil_direction                  = kwargs['coil_direction']
        coil_diameter                   = kwargs.get('coil_diameter', 100)
        # Calculate all the distance
        CoilCenterTensor, _, _, _       = np.meshgrid(
            coil_center,
            self._cellCenters,
            self._cellCenters,
            self._cellCenters,
            indexing='ij'
        )
        CoilDirectionTensor, _, _, _    = np.meshgrid(
            coil_direction,
            self._cellCenters,
            self._cellCenters,
            self._cellCenters,
            indexing='ij'
        )
        Distance                        = self._cellCenterTensor - CoilCenterTensor
        AbsDistance                     = np.sqrt(
            np.sum((self._cellCenterTensor - CoilCenterTensor) ** 2, axis=0)
        )
        # # debug
        # print("CoilDirectionTensor shape = "+str(CoilDirectionTensor.shape))
        # print("Distance shape = "+str(Distance.shape))
        ProductDirection                = np.abs(np.sum(
            CoilDirectionTensor * Distance,
            axis=0
        ) / AbsDistance)
        return self._findOptimalSurfaceInds(ProductDirection, AbsDistance, coil_diameter=coil_diameter)

    def _findOptimalSurfaceInds(self, ProductDirection, AbsDistance, **kwargs):
        # load info
        coil_diameter           = kwargs.get('coil_diameter', 100)
        ###############
        xinds_array             = []
        yinds_array             = []
        zinds_array             = []
        lengths                 = []
        for axis in [0, 1, 2]:
            # Find the minimum along Z (required axis)
            zinds = np.argmin(ProductDirection, axis=axis)
            # # debug
            # print("zinds = "+str(zinds))
            xinds, yinds = np.meshgrid(
                np.linspace(0, self._numCells - 1, self._numCells).astype(np.int),
                np.linspace(0, self._numCells - 1, self._numCells).astype(np.int),
                indexing='ij'
            )
            # Find the one with distance < r
            if axis == 0:
                inds1, inds2            = np.where(
                    AbsDistance[zinds, xinds, yinds] < 0.5 * coil_diameter
                )
            elif axis == 1:
                inds1, inds2            = np.where(
                    AbsDistance[xinds, zinds, yinds] < 0.5 * coil_diameter
                )
            else:
                inds1, inds2            = np.where(
                    AbsDistance[xinds, yinds, zinds] < 0.5 * coil_diameter
                )
            # append
            lengths.append(int(len(inds1)))
            xinds_array.append(deepcopy(
                xinds[inds1, inds2]
            ))
            yinds_array.append(deepcopy(
                yinds[inds1, inds2]
            ))
            zinds_array.append(deepcopy(
                zinds[inds1, inds2]
            ))
        # check
        index = np.argmax(lengths)
        return xinds_array[index], yinds_array[index], zinds_array[index]


    ###################################################
    ## Public functions
    ###################################################
    def getPartType(self):
        return self._partType

    def getBoxSize(self):
        return self._worldBoxSize

    def calcInductionVoltage(self, **kwargs):
        '''
        Calculate the induction voltage based on input B field template
        :param kwargs:
        :return:
        '''
        # load info
        part_start_point                = kwargs.get('start_point', np.zeros(3))
        part_direction                  = kwargs.get('direction', np.asarray([0,0,1]))
        coil_center                     = kwargs.get('coil_center', np.zeros(3))
        coil_direction                  = kwargs.get('coil_direction', np.asarray([0,0,1]))
        coil_diameter                   = kwargs.get('coil_diameter', 100)
        part_speed                      = kwargs.get('part_speed', 0.001)
        part_electric_charge            = kwargs.get('part_electric_charge', 1)
        part_magnetic_charge            = kwargs.get('part_magnetic_charge', 1)
        part_magnetic_moment            = kwargs.get('part_magnetic_moment', [0,0,9.284e-24])
        # transfer the coordinates so that particle moves along Z+
        # and coil center on X+
        coil_center_prime, coil_direction_prime     = self._transferCoordinates(
            part_start_point            = part_start_point,
            part_direction              = part_direction,
            coil_center                 = coil_center,
            coil_direction              = coil_direction,
        )
        # get the index of cells that will be used to calculate B
        xinds, yinds, zinds             = self._getCoilSurfaceIndex(
            coil_center                 = coil_center_prime,
            coil_direction              = coil_direction_prime,
            coil_diameter               = coil_diameter,
        )
        # calculate B flux
        Delta_B                         = self._B[1:,:,:,:,:] - self._B[:-1,:,:,:,:]
        Delta_t                         = self._particleStep / (self._speedOfLight*self._partSpeed) # in ns
        BFlux                           = np.sum(
            Delta_B[:, :, xinds, yinds, zinds]*self._cellStep**2,
            axis                        =1
        )
        # convert to right unit
        BFlux                          *= 1.e-15 # V/s
        # get the voltages
        Vs                              = -BFlux/Delta_t*1e-9 # in V
        # scale the voltages due to the different particle properties
        Vs                             *= part_speed/self._partSpeed
        if self._partType==0:
            Vs                         *= part_magnetic_charge / self._partMagneticCharge
        elif self._partType==1:
            Vs                         *= part_electric_charge / self._partElectricCharge
        elif self._partType==2:
            index                       = np.where(self._partMagneticMoment!=0)[0]
            if len(index)>1:
                raise ValueError("Template has magnetic moment not on X or Y or Z direction")
            Vs                         *= part_magnetic_moment[index[0]] / self._partMagneticMoment[index[0]]
        return np.asarray(Vs)*self._numCoilTurns


##########################################
## Induction calculator
## using multiple B field templates instead of a single one
##########################################

class MultipleInductionCalculator:
    def __init__(self, **kwargs):
        # load one by one
        self._calculators               = []
        for config in kwargs['configs']:
            self._calculators.append(
                SingleInductionCalculator(**config)
            )
        # check part_type & box size
        part_types              = []
        box_sizes               = []
        for calculator in self._calculators:
            part_types.append(calculator.getPartType())
            box_sizes.append(calculator.getBoxSize())
        if len(np.unique(part_types))<1 or len(np.unique(box_sizes))<1:
            raise ValueError("No config is given for Multiple induction calculator!")
        elif len(np.unique(part_types))>1 or len(np.unique(box_sizes))>1:
            raise ValueError("Input configs for multiple induction are with different particle types!")

        return

    #########################################
    ## Public functions
    #########################################
    def getPartType(self):
        return self._calculators[0].getPartType()

    def getBoxSize(self):
        return self._calculators[0].getBoxSize()

    def calcInductionVoltage(self, **kwargs):
        Vs                  = None
        for index, calculator in enumerate(self._calculators):
            if index==0:
                Vs          = calculator.calcInductionVoltage(**kwargs)
            else:
                Vs         += calculator.calcInductionVoltage(**kwargs)
        return Vs

