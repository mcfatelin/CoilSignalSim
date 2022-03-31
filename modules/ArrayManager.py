#######################################################
## Array Manager manages the coils and their positions
## by Qing Lin @ 2021-12-07
#######################################################
import numpy as np
import pickle as pkl




class ArrayManager:
    def __init__(self, **kwargs):
        self._coilDiameter          = kwargs.get('coil_diameter', 100)
        self._loadCoils(kwargs['coil_arrange_filename'])
        return

    ####################################
    ## Private functions
    ####################################
    def _loadCoils(self, filename):
        '''
        Load an array of dictionary
        each dictionary gives the center and direction (3) array
        :param filename:
        :return:
        '''
        # load the config file
        Array                       = pkl.load(open(filename, 'rb'))
        # loop over and load the centers and directions of the coils
        self._coilIDs               = []
        self._coilCenters           = []
        self._coilDirections        = []
        for Dict in Array:
            self._coilIDs.append(Dict['coil_id'])
            self._coilCenters.append(np.asarray(Dict['coil_center']))
            self._coilDirections.append(np.asarray(Dict['coil_direction']))
        self._coilIDs               = np.asarray(self._coilIDs)
        self._coilCenters           = np.asarray(self._coilCenters)
        self._coilDirections        = np.asarray(self._coilDirections)
        return




    ####################################
    ## Public functions
    #####################################
    def calcHitCoils(self, **kwargs):
        '''
        Find which coils are "hit".
        If the shortest distance between particle and coil center < coil diameter
        we consider it an "hit"
        :return:
        '''
        # load info
        part_start_point            = kwargs.get('part_start_point', np.asarray([0,0,0]))
        part_direction              = kwargs.get('part_direction', np.asarray([0,0,1]))
        part_speed                  = kwargs.get('part_speed', 2.98e2) #mm/ns
        tolerance                   = kwargs.get('tolerance', self._coilDiameter)
        tolerance                  -= self._coilDiameter/2.
        # debug
        # print("===>>> Part. start point = "+str(part_start_point))
        # print("===>>> Part. direction = "+str(part_direction))
        # normalize the direction
        part_direction              /= np.sqrt(np.sum(np.power(part_direction,2.)))
        # make the start_point to (N,3)
        _, part_start_point,        = np.meshgrid(
            np.zeros(self._coilIDs.shape[0]),
            part_start_point,
            indexing                = 'ij'
        )
        _, part_direction,          = np.meshgrid(
            np.zeros(self._coilIDs.shape[0]),
            part_direction,
            indexing                = 'ij'
        )
        # distance matrix
        distance_matrix             = self._coilCenters - part_start_point
        abs_distance                = np.sqrt(np.sum(distance_matrix**2, axis=1))
        # costheta
        costheta                    = np.sum(part_direction*distance_matrix, axis=1) / abs_distance
        sintheta                    = np.sqrt(1. -  costheta**2)
        # minimal distance
        min_distance                = abs_distance*sintheta
        # find the hits
        inds                        = np.where(min_distance<tolerance)[0]
        self._hitCoilIDs            = self._coilIDs[inds]
        self._hitCoilCenters        = self._coilCenters[inds]
        self._hitCoilDirections     = self._coilDirections[inds]
        self._hitTimes              = abs_distance*costheta / part_speed # in ns
        abs_multi_costheta, _       = np.meshgrid(
            abs_distance*costheta,
            np.zeros(part_direction.shape[1]),
            indexing                = 'ij'
        )
        self._hitStartPoints        = part_start_point + part_direction*abs_multi_costheta
        return

    def getHits(self):
        # make an array of dictionary
        Array               = []
        for coil_id, coil_center, coil_direction, hit_time, hit_start_point in zip(
            self._hitCoilIDs,
            self._hitCoilCenters,
            self._hitCoilDirections,
            self._hitTimes,
            self._hitStartPoints,
        ):
            Dict                    = {}
            Dict['coil_id']         = coil_id
            Dict['coil_center']     = coil_center
            Dict['coil_direction']  = coil_direction
            Dict['hit_time']        = hit_time
            Dict['hit_start_point'] = hit_start_point
            Array.append(Dict)
        return Array

    def getCoilIDs(self):
        return self._coilIDs
