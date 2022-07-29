import pickle as pkl
import numpy
Output_Dict = {}
Output_Dict['consider_freq'] = 40000*1e-6 # in Hz
Output_Dict['T2']            = 0.0002 # in s
pkl.dump(
    Output_Dict,
    open("Magnetormetertest.pkl", 'wb')
    )