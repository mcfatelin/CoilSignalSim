##########################################
## Main code for simulation
## by Qing Lin
## @ 2021-11-18
##########################################
import sys



#########################
## Input
#########################
if len(sys.argv)<2:
    Message     = 'python3 RunSim.py <config file> <number of sim> < (opt.) output filename> <(opt.)Save type>\n'
    Message     += 'Save type = 0 saves only the coil Voltage waveform\n'
    Message     += 'Save type = 1 saves also the field tensor for each particle step (Note files will be large)'
    print(Message)
    exit()


ConfigFilename          = sys.argv[1]
NumEvents               = int(sys.argv[2])
OutputFilename          = None
if len(sys.argv)>3:
    OutputFilename      = sys.argv[3]
SaveType                = 0
if len(sys.argv)>4:
    SaveType            = int(sys.argv[4])


############################
## Initiate master controller
############################
from modules.MasterHandler import MasterHandler

handler                 = MasterHandler(
    config_filename     = ConfigFilename
)



##############################
## Run Simulation
##############################

handler.run_sim(
    num_sim             = NumEvents,
    output_filename     = OutputFilename,
    save_type           = SaveType,
)
