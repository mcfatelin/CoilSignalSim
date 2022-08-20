### 
import numpy as np
class ThresholdTrigger():
    def __init__(self,**kwargs):
        self.threshold = kwargs.get('threshold',0.9)
        self.threshold *= 1e8
    def _calculate(self,voltages):
        if np.max(np.abs(voltages)) > self.threshold:
                hits = 1
                hit_intensity = np.max(np.abs(voltages))
        else:
                hits = 0
                hit_intensity = 0
        return hits,hit_intensity