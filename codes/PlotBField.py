#########################################
# Plot the field and voltage
# and make them into a gif
# by Qing Lin
# @ 2021-11-23
#########################################
import pickle as pkl
import numpy as np
import sys

SpeedOfLight                = 2.998e8 # m/s

#################################
## Inputs
#################################
if len(sys.argv)<2:
    print("python3 PlotBField.py <input File> <output dir>")
    exit()

InputFilename               = sys.argv[1]
OutputDir                   = sys.argv[2]


################################
## Load
################################
Dict                        = pkl.load(open(InputFilename, 'rb'))


if 'B_tensors' not in Dict.keys():
    raise ValueError("The input file has a save_type==0, not compatible with this plotting code!")


################################
## Loop over and Plot
###############################
from tqdm import tqdm
from Tools import PlotSingleBField


B_tensor                    = np.asarray(Dict['B_tensors'])

NumSteps                     = int(len(Dict['B_tensors']))


B_filelist                      = []

Bx_mean                         = np.average(B_tensor[:,0,:,:,:])
Bx_std                          = np.std(B_tensor[:,0,:,:,:])
Bx_min                          = Bx_mean - 10*Bx_std
Bx_max                          = Bx_mean + 10*Bx_std

By_mean                         = np.average(B_tensor[:,1,:,:,:])
By_std                          = np.std(B_tensor[:,1,:,:,:])
By_min                          = By_mean - 10*By_std
By_max                          = By_mean + 10*By_std

Bz_mean                         = np.average(B_tensor[:,2,:,:,:])
Bz_std                          = np.std(B_tensor[:,2,:,:,:])
Bz_min                          = Bz_mean - 10*Bz_std
Bz_max                          = Bz_mean + 10*Bz_std




for ii in tqdm(range(NumSteps)):
    # plot B field
    filename            = OutputDir+'/B_field_step'+str(ii)+'.jpg'
    PlotSingleBField(
        Dict['B_tensors'],
        output_filename         =filename,
        step_index              =ii,
        EorB                    =1,
        part_speed              =Dict['part_speed'],
        comp_mins               =[Bx_min, By_min, Bz_min],
        comp_maxs               =[Bx_max, By_max, Bz_max],
        config                  =Dict['config'],
    )
    B_filelist.append(filename)



#######################
## Make gif
#######################
from Tools import MakeGif


MakeGif(
    filelist                = B_filelist,
    output_filename         = OutputDir+'/B_field.gif',
    duration                = 0.1
)



