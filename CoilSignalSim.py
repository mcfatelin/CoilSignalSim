#############################################
## Main code for coil-based Monopole detector simulation,
## simulating the signal pulse induced by particle passing
## This is a code re-arrangment compared to previous version
## by Qing Lin
## @ 2021-12-02
##############################################
import sys


################################
## Input
################################
if len(sys.argv)<2:
    Message             = "python3 CoilSignalSim.py <Mode> <config file> <output filename> <(opt.)Number of simulated events>"
    Message            += "Mode==0 is the field generation mode."
    Message            += "Mode==1 is the induction signal calculation mode."
    print(Message)
    exit()


Mode                        = int(sys.argv[1])
ConfigFilename              = sys.argv[2]
OutputFilename              = sys.argv[3]
NumEvents                   = 1
if len(sys.argv)>4:
    NumEvents               = int(sys.argv[4])


################################
## Initiate the controller
################################
from modules.FieldGenerator import FieldGenerator
from modules.SignalCalculator import SignalCalculator

handler                 = None
if Mode==0:
    handler             = FieldGenerator(
        config_filename = ConfigFilename
    )
else:
    handler             = SignalCalculator(
        config_filename = ConfigFilename
    )


###################################
## Run simulation
###################################


handler.run_sim(
    num_sim                 = NumEvents,
    output_filename         = OutputFilename,
)