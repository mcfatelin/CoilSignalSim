########################################
## Make a single coil arrangement
## with a coil surface covering cubic surface
## the cubic is with a center at (0,0,0)
########################################
import numpy as np
import pickle as pkl
from copy import deepcopy


Array               = []
radius = 20
Array.append(
    dict(
        coil_id         = 0,
        coil_center     = [radius,0,0],
        coil_direction  = [-1,0,0],
    )
)

Array.append(
    dict(
        coil_id         = 1,
        coil_center     = [-radius,0,0],
        coil_direction  = [1,0,0],
    )
)

Array.append(
    dict(
        coil_id         = 2,
        coil_center     = [0,radius,0],
        coil_direction  = [0,-1,0],
    )
)

Array.append(
    dict(
        coil_id         = 3,
        coil_center     = [0,-radius,0],
        coil_direction  = [0,1,0],
    )
)

Array.append(
    dict(
        coil_id         = 4,
        coil_center     = [0,0,radius],
        coil_direction  = [0,0,-1],
    )
)

Array.append(
    dict(
        coil_id         = 5,
        coil_center     = [0,0,-radius],
        coil_direction  = [0,0,1],
    )
)


pkl.dump(
    Array,
    open("single_coil_array2.pkl", 'wb')
)




