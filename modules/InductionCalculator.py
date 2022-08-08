#########################################################
## Module for calculating the amount of voltages induced by Magnetic field derivative
## by Qing Lin @ 2021-12-07
## Adding an option to use analytic calculator originally written by Xiang Kang
## by Qing Lin @ 2022-03-28
##########################################################
import numpy as np
import pickle as pkl
from copy import deepcopy
from scipy import special




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

    def getParticleStep(self):
        return self._calculator.getParticleStep()

    def getNumberOfSamples(self):
        return self._calculator.getNumberOfSamples()

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
        self._diracMagneticCharge   = 4.14125e-15 # Wb
        return


    def _loadConfig(self, **kwargs):
        '''
        Load info from a file
        :param kwargs:
        :return:
        '''
        # load info
        self._numCoilTurns              = kwargs.get('num_coil_turns', 4000)
        self._usingAnalytic             = kwargs.get('using_analytic', True)
        if not self._usingAnalytic:
            ##############################
            # Load simulated field
            ##############################
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
        else:
            ##############################
            # Load analytic calculator
            # analytic method is based on sample times
            # NOTE currently analytic method is only available for monopole
            ##############################
            # get information
            self._worldBoxSize              = kwargs.get('world_box_size', 1e5) # unit in mm
            self._distanceBeforeZero        = kwargs.get('distance_before_zero', 500) # unit in mm, distance before reaching the closest point with respect to the coil center
            self._numSamples                = kwargs.get('num_samples', 2000)
            self._partType                  = 0 # hard coded, currently analytic method is only applicable to monopole calculation
            self._particleStep              = self._worldBoxSize / float(self._numSamples - 1)
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
        elif axis==2:
            rotation_matrix = np.asarray([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), +np.cos(angle), 0],
                [0,0,1.]
            ])
        else:
            pass
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
        return self._findOptimalSurfaceInds(
            ProductDirection,
            AbsDistance,
            coil_diameter           = coil_diameter,
            coil_direction          = coil_direction,
        )

    def _findOptimalSurfaceInds(self, ProductDirection, AbsDistance, **kwargs):
        # load info
        coil_diameter           = kwargs.get('coil_diameter', 100)
        coil_direction          = kwargs['coil_direction']
        ###############
        # find the axis for projection
        ###############
        xinds_array = []
        yinds_array = []
        zinds_array = []
        lengths = []
        # axis                    = np.argmax(coil_direction)
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
            if axis==0:
                xinds_array.append(deepcopy(
                    zinds[inds1, inds2]
                ))
                yinds_array.append(deepcopy(
                    xinds[inds1, inds2]
                ))
                zinds_array.append(deepcopy(
                    yinds[inds1, inds2]
                ))
            elif axis==1:
                xinds_array.append(deepcopy(
                    xinds[inds1, inds2]
                ))
                yinds_array.append(deepcopy(
                    zinds[inds1, inds2]
                ))
                zinds_array.append(deepcopy(
                    yinds[inds1, inds2]
                ))
            else:
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

    def getParticleStep(self):
        return self._particleStep

    def getNumberOfSamples(self):
        if self._usingAnalytic:
            return self._numSamples
        else:
            return self._B.shape[0]

    #########################################
    ## Two methods for calculating induction voltages
    #########################################
    def calcInductionVoltageFiniteElement(self, **kwargs):
        '''
        Calculate the induction voltage based on input B field template, produced using finite element method.
        :param kwargs:
        :return:
        '''
        # load info
        part_start_point = kwargs.get('start_point', np.zeros(3))
        part_direction = kwargs.get('direction', np.asarray([0, 0, 1]))
        coil_center = kwargs.get('coil_center', np.zeros(3))
        coil_direction = kwargs.get('coil_direction', np.asarray([0, 0, 1]))
        coil_diameter = kwargs.get('coil_diameter', 100)
        part_speed = kwargs.get('part_speed', 0.001)
        part_electric_charge = kwargs.get('part_electric_charge', 1)
        part_magnetic_charge = kwargs.get('part_magnetic_charge', 1)
        part_magnetic_moment = kwargs.get('part_magnetic_moment', [0, 0, 9.284e-24])
        # transfer the coordinates so that particle moves along Z+
        # and coil center on X+
        # # debug
        # print("coil_center = "+str(coil_center))
        # print("coil_direction = "+str(coil_direction))
        coil_center_prime, coil_direction_prime = self._transferCoordinates(
            part_start_point=part_start_point,
            part_direction=part_direction,
            coil_center=coil_center,
            coil_direction=coil_direction,
        )
        # # debug
        # print("coil_center_prime = "+str(coil_center_prime))
        # print("coil_direction_prime = "+str(coil_direction_prime))
        # get the index of cells that will be used to calculate B
        xinds, yinds, zinds = self._getCoilSurfaceIndex(
            coil_center=coil_center_prime,
            coil_direction=coil_direction_prime,
            coil_diameter=coil_diameter,
        )
        # debug
        # print("xinds = "+str(xinds))
        # print("yinds = "+str(yinds))
        # print("zinds = "+str(zinds))
        # calculate B flux
        Delta_B = self._B[1:, :, :, :, :] - self._B[:-1, :, :, :, :]
        Delta_t = self._particleStep / (self._speedOfLight * self._partSpeed)  # in ns
        BFlux = np.sum(
            Delta_B[:, :, xinds, yinds, zinds] * self._cellStep ** 2,
            axis=(1, 2)
        )
        # print("BFlux shape = "+str(BFlux.shape))
        # convert to right unit
        BFlux *= 1.e-15  # V/s
        # get the voltages
        Vs = np.append(
            np.zeros(1),
            -BFlux / Delta_t * 1e9  # in V
        )
        # scale the voltages due to the different particle properties
        Vs *= part_speed / self._partSpeed
        if self._partType == 0:
            Vs *= part_magnetic_charge / self._partMagneticCharge
        elif self._partType == 1:
            Vs *= part_electric_charge / self._partElectricCharge
        elif self._partType == 2:
            index = np.where(self._partMagneticMoment != 0)[0]
            if len(index) > 1:
                raise ValueError("Template has magnetic moment not on X or Y or Z direction")
            Vs *= part_magnetic_moment[index[0]] / self._partMagneticMoment[index[0]]
        # calculate the sample times
        sample_times = np.linspace(
            -0.5 * self.getBoxSize() / part_speed / self._speedOfLight,
            +0.5 * self.getBoxSize() / part_speed / self._speedOfLight,
            self.getNumberOfSamples()
        )
        return (np.asarray(Vs) * self._numCoilTurns, sample_times)


    def calcInductionVoltageAnalytic(self, **kwargs):
        '''
        Calculate the induction voltage based on analytic method.
        Note currently this method in ONLY applicable to monopole-related calculation.
        Calculation copy right @Xiang Kang
        :param kwargs:
        :return:
        '''
        #####################
        # load info
        #####################
        part_start_point                = kwargs.get('start_point', np.zeros(3))
        part_direction                  = kwargs.get('direction', np.asarray([0, 0, 1]))
        coil_center                     = kwargs.get('coil_center', np.zeros(3))
        coil_direction                  = kwargs.get('coil_direction', np.asarray([0, 0, 1]))
        coil_diameter                   = kwargs.get('coil_diameter', 100)
        part_speed                      = kwargs.get('part_speed', 0.001)
        part_magnetic_charge            = kwargs.get('part_magnetic_charge', 1)
        #####################
        # Get the derivative parameters
        #####################
        t                               = np.linspace(
            0.,
            self._worldBoxSize / part_speed / self._speedOfLight,
            int(self._numSamples)
        )  # in ns
        direction_theta, direction_phi, start_rho, start_z            = self.getParInCoilCylindricalCoord(
            part_start_point,
            part_direction,
            coil_center,
            coil_direction,
            self._distanceBeforeZero,
        ) # all in mm
        t                              *= 1e-9 # in s
        #####################
        # Convert to another unit system
        # to make the equations readable and simpler
        #####################
        theta                   = direction_theta                                       # rad
        phi                     = direction_phi                                         # rad
        coil_radius             = coil_diameter / 2. * 1e-3                             # m
        vN                      = part_speed * self._speedOfLight * 1e6 / coil_radius   # m/s
        start_rhoN              = start_rho*1e-3 / coil_radius                          # m/m
        start_zN                = -start_z*1e-3 / coil_radius                            # m/m
        #####################
        # calculate rho/coil_radius and Z/radius value for Ts
        #####################
        rhoN                    = np.sqrt((vN*t*np.sin(theta))**2+start_rhoN**2+2*vN*t*start_rhoN*np.sin(theta)*np.cos(phi))
        zN                      = vN*t*np.cos(theta)-start_zN
        #####################
        # calculate the time differential of solid angle
        #####################
        if theta==0:
            # corresponding to the case where particle passing coil plane perpendicularly
            factor              = 2 * vN / (zN ** 2 + (rhoN - 1) ** 2) / np.sqrt(zN ** 2 + (rhoN + 1) ** 2)
            part1               = -(zN ** 2 + (rhoN - 1) ** 2) * special.ellipk(4 * rhoN / (zN ** 2 + (1 + rhoN) ** 2))
            part2               = (zN ** 2 + rhoN ** 2 - 1) * special.ellipe(4 * rhoN / (zN ** 2 + (1 + rhoN) ** 2))
            omegaT              = factor*(part1+part2)
            # omegaT[0]           = 0 # to avoid nan, hardcoding
        elif start_rhoN==0:
            # t==0 would be none calculatable
            omegaT_0            = -2*np.pi*np.sqrt(1+zN**2)*vN*np.cos(theta)/(zN**2+1)**2
            factor              = 2 / (zN ** 2 + (1 - rhoN) ** 2) / rhoN / np.sqrt(zN ** 2 + (1 + rhoN) ** 2)
            rhoT                = (vN * start_rhoN * np.cos(phi) * np.sin(theta) + vN ** 2 * t * np.sin(theta) ** 2) / rhoN
            zT                  = vN * np.cos(theta)
            part1               = (zN ** 2 + (1 - rhoN) ** 2) * (zN * rhoT - rhoN * zT) * special.ellipk(4 * rhoN / (zN ** 2 + (1 + rhoN) ** 2))
            part2               = (zT * rhoN * (rhoN ** 2 + zN ** 2 - 1) - zN * (1 + zN ** 2 + rhoN ** 2) * rhoT) * special.ellipe(4 * rhoN / (zN ** 2 + (1 + rhoN) ** 2))
            omegaT              = factor * (part1 + part2)
            omegaT[0]           = omegaT_0
        else:
            # corresponding to general case
            factor              = 2 / (zN ** 2 + (1 - rhoN) ** 2) / rhoN / np.sqrt(zN ** 2 + (1 + rhoN) ** 2)
            rhoT                = (vN * start_rhoN * np.cos(phi) * np.sin(theta) + vN ** 2 * t * np.sin(theta) ** 2) / rhoN
            zT                  = vN * np.cos(theta)
            part1               = (zN ** 2 + (1 - rhoN) ** 2) * (zN * rhoT - rhoN * zT) * special.ellipk(4 * rhoN / (zN ** 2 + (1 + rhoN) ** 2))
            part2               = (zT * rhoN * (rhoN ** 2 + zN ** 2 - 1) - zN * (1 + zN ** 2 + rhoN ** 2) * rhoT) * special.ellipe(4 * rhoN / (zN ** 2 + (1 + rhoN) ** 2))
            omegaT = factor * (part1 + part2)
        ######################
        ## calculate the induction voltage
        ## and the sample_times
        ######################
        em                      = -float(self._numCoilTurns)*self._diracMagneticCharge*part_magnetic_charge/4./np.pi*omegaT
        closest_point_time      = self._distanceBeforeZero / part_speed / self._speedOfLight # in ns
        sample_times            = t*1e9 - closest_point_time # in ns
        return (em, sample_times)


    def getParInCoilCylindricalCoord(self, part_start_point, part_direction, coil_center, coil_direction, distance_before_zero):
        '''
        Calculate the direction_theta, direction_phi, new_start_rho par for analytic method
        Also calculate the distance between closest point with respect to coil center to the start point
        :param part_start_point:
        :param part_direction:
        :param coil_center:
        :param coil_direction:
        :param new_start_Z: Note this is under transferred coordinates, not the same as above
        :return:
        '''
        # to numpy
        part_start_point            = np.asarray(part_start_point).astype(np.float)
        part_direction              = np.asarray(part_direction).astype(np.float)
        coil_center                 = np.asarray(coil_center).astype(np.float)
        coil_direction              = np.asarray(coil_direction).astype(np.float)
        # normalize
        part_direction              /= np.sqrt(np.sum(part_direction**2))
        coil_direction              /= np.sqrt(np.sum(coil_direction**2))
        #######################
        # Translate until coil_center is at 000
        #######################
        # # debug
        # print("=================")
        # print("Before translation:")
        # print("===>>> Coil center = "+str(coil_center))
        # print("===>>> Coil Direction = "+str(coil_direction))
        # print("===>>> Part. start point = "+str(part_start_point))
        # print("===>>> Part. direction = "+str(part_direction))
        # print()
        translation                 = -coil_center
        part_start_point            = self._translate(part_start_point, translation)
        coil_center                 = self._translate(coil_center, translation)
        #######################
        ## rotate until coil_direction is on YZ plane
        #######################
        # # debug
        # print("Before coil direction rotation along Z+:")
        # print("===>>> Coil center = "+str(coil_center))
        # print("===>>> Coil Direction = "+str(coil_direction))
        # print("===>>> Part. start point = "+str(part_start_point))
        # print("===>>> Part. direction = "+str(part_direction))
        # print()
        angle                       = np.pi/2. - np.arctan2(coil_direction[1], coil_direction[0])
        part_start_point, part_direction            = self._rotate(part_start_point, part_direction, angle=angle, axis=2)
        coil_center, coil_direction                 = self._rotate(coil_center, coil_direction, angle=angle, axis=2)
        #######################
        # Rotate until coil direction is on Z+
        #######################
        # # debug
        # print("Before coil direction rotation along X+:")
        # print("===>>> Coil center = "+str(coil_center))
        # print("===>>> Coil Direction = "+str(coil_direction))
        # print("===>>> Part. start point = "+str(part_start_point))
        # print("===>>> Part. direction = "+str(part_direction))
        # print()
        angle                       = np.pi/2. - np.arctan2(coil_direction[2], coil_direction[1])
        part_start_point, part_direction            = self._rotate(part_start_point, part_direction, angle=angle, axis=0)
        coil_center, coil_direction                 = self._rotate(coil_center, coil_direction, angle=angle, axis=0)
        #######################
        # Rotate until start_point has Y=0
        #######################
        # # debug
        # print("Before start point rotation to Y=0:")
        # print("===>>> Coil center = "+str(coil_center))
        # print("===>>> Coil Direction = "+str(coil_direction))
        # print("===>>> Part. start point = "+str(part_start_point))
        # print("===>>> Part. direction = "+str(part_direction))
        # print("=================")
        if part_start_point[0]**2+part_start_point[1]**2<0.0001:
            # if start point XY is too close to 0
            # shift it a bit
            part_start_point            += 10.*part_direction
        angle                                       = np.pi/2. - np.arctan2(part_start_point[1], part_start_point[0])
        part_start_point, part_direction            = self._rotate(part_start_point, part_direction, angle=angle, axis=2)
        coil_center, coil_direction                 = self._rotate(coil_center, coil_direction, angle=angle, axis=2)
        #######################
        # calculate outputs
        #######################
        direction_theta                             = np.pi/2. - np.arctan2(part_direction[2], np.sqrt(part_direction[0]**2+part_direction[1]**2))
        direction_phi                               = np.arctan2(part_direction[1], part_direction[0])
        if direction_phi<0:
            direction_phi                          += 2.*np.pi
        #######################
        # Calculate the closest point
        #######################
        closest_distance_from_start                 = -np.sum(part_start_point * part_direction) / np.sum(part_direction ** 2)
        #######################
        # Re-define the start point to match customized start_z
        #######################
        # # debug
        # print("Before redefine part. start point:")
        # print("===>>> Coil center = "+str(coil_center))
        # print("===>>> Coil Direction = "+str(coil_direction))
        # print("===>>> Part. start point = "+str(part_start_point))
        # print("===>>> Part. direction = "+str(part_direction))
        # print()
        part_start_point                            += (closest_distance_from_start - distance_before_zero)*part_direction
        new_start_z                                 = part_start_point[2]
        new_start_rho                               = np.sqrt(part_start_point[0]**2+part_start_point[1]**2)
        # # debug
        # print("Outputs:")
        # print("===>>> Direction theta = "+str(direction_theta))
        # print("===>>> Direction phi = "+str(direction_phi))
        # print("===>>> start_rho = "+str(new_start_rho))
        # print("===>>> start_z = "+str(new_start_z))
        # print()
        return (
            direction_theta,
            direction_phi,
            new_start_rho,
            new_start_z,
        )


    #########################################
    #########################################


    def calcInductionVoltage(self, **kwargs):
        '''
        Calculate the induction voltage based on input B field template
        :param kwargs:
        :return:
        '''
        if self._usingAnalytic:
            return self.calcInductionVoltageAnalytic(**kwargs)
        else:
            return self.calcInductionVoltageFiniteElement(**kwargs)


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

    def getParticleStep(self):
        return self._calculators[0].getParticleStep()

    def getNumberOfSamples(self):
        return self._calculators[0].getNumberOfSamples()

    def calcInductionVoltage(self, **kwargs):
        Vs                  = None
        sample_times        = None
        for index, calculator in enumerate(self._calculators):
            if index==0:
                Obj             = calculator.calcInductionVoltage(**kwargs)
                sample_times    = Obj[-1]
                Vs              = Obj[0]
            else:
                Obj             = calculator.calcInductionVoltage(**kwargs)
                Vs             += Obj[0]
        return (Vs, sample_times)

