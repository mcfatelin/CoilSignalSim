#################################################################################
## Module for calculating the RLC circuit response
## Effective serial and parallel branches are okay, but tree-level or loop-level are not supported
## Data structure use "tree" structure to store effective L, R, C, and I, U
## by Qing Lin @ 2021-12-23
#################################################################################
import numpy as np
import pickle as pkl



class CircuitCalculator:
    # place-holder
    def __init__(self, **kwargs):
        return

    ########################################
    ## Private functions
    ########################################



    #######################################
    ## Public functions
    #######################################
    def getSamplingTimes(self):
        # placeholder
        return np.zeros(10)

    def calculateVoltages(self, **kwargs):
        # placeholder
        return np.zeros(10)



