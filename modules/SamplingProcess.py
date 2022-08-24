# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 14:36:51 2022

@author: admin
"""

import numpy as np
from scipy.interpolate import interp1d
class SamplingProcess():
    def __init__(self,t,induction_voltages):
        self._interpolate(t,induction_voltages)
        return
    def _interpolate(self,t,induction_voltages):
        self.t = t
        self.voltages = induction_voltages
        self._interInduction = interp1d(
                                        t,
                                        induction_voltages,
                    bounds_error        = False,
                    fill_value          = (induction_voltages[0], induction_voltages[-1]),
                                        )
    def sampling(self,**kwargs):
        sample_rate  = kwargs.get('sample_rate',10000000)  ## in Hz
        t_range = max(self.t) - min(self.t)
        #print(t_range)
        number_of_samples  = int(t_range*sample_rate*1e-9)
        #print(number_of_samples)
        sampling_t                  = np.linspace(min(self.t),max(self.t),number_of_samples) ## in ns
        #print(sampling_t)
        sampling_induction = self._interInduction(sampling_t)
        if len(sampling_induction) == 0:
            #print('self.t and self.voltages',self.t,self.voltages,'\n')
            return (sampling_t,self.voltages)
        return (sampling_t,sampling_induction)