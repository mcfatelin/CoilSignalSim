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
    print("python3 PlotField.py <input File> <output dir> <Event Number (default 0)>")
    exit()

InputFilename               = sys.argv[1]
OutputDir                   = sys.argv[2]
EventNum                    = 0
if len(sys.argv)>3:
    EventNum                = int(sys.argv[3])


################################
## Load
################################
Dict                        = pkl.load(open(InputFilename, 'rb'))


if ('E_tensors' not in Dict.keys()) or ('B_tensors' not in Dict.keys()):
    raise ValueError("The input file has a save_type==0, not compatible with this plotting code!")

if EventNum>=len(Dict['E_tensors']):
    raise ValueError("Requested event number is not available!")

################################
## Loop over and Plot
###############################
from tqdm import tqdm
from Tools import PlotSingleField

E_tensor                    = np.asarray(Dict['E_tensors'][EventNum])
B_tensor                    = np.asarray(Dict['B_tensors'][EventNum])

NumSteps                     = int(len(Dict['E_tensors'][EventNum]))

E_filelist                      = []
B_filelist                      = []

Ex_mean                         = np.average(E_tensor[:,0,:,:,:])
Ex_std                          = np.std(E_tensor[:,0,:,:,:])
Ex_min                          = Ex_mean - 10*Ex_std
Ex_max                          = Ex_mean + 10*Ex_std

Ey_mean                         = np.average(E_tensor[:,1,:,:,:])
Ey_std                          = np.std(E_tensor[:,1,:,:,:])
Ey_min                          = Ey_mean - 10*Ey_std
Ey_max                          = Ey_mean + 10*Ey_std

Ez_mean                         = np.average(E_tensor[:,2,:,:,:])
Ez_std                          = np.std(E_tensor[:,2,:,:,:])
Ez_min                          = Ez_mean - 10*Ez_std
Ez_max                          = Ez_mean + 10*Ez_std

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



# Ex_min                          = np.min(E_tensor[:,0,:,:,:])
# Ey_min                          = np.min(E_tensor[:,1,:,:,:])
# Ez_min                          = np.min(E_tensor[:,2,:,:,:])
# Ex_max                          = np.max(E_tensor[:,0,:,:,:])
# Ey_max                          = np.max(E_tensor[:,1,:,:,:])
# Ez_max                          = np.max(E_tensor[:,2,:,:,:])
# 
# Bx_min                          = np.min(B_tensor[:,0,:,:,:])
# By_min                          = np.min(B_tensor[:,1,:,:,:])
# Bz_min                          = np.min(B_tensor[:,2,:,:,:])
# Bx_max                          = np.max(B_tensor[:,0,:,:,:])
# By_max                          = np.max(B_tensor[:,1,:,:,:])
# Bz_max                          = np.max(B_tensor[:,2,:,:,:])




for ii in tqdm(range(NumSteps)):
    # plot E field
    filename            = OutputDir+'/E_field_step'+str(ii)+'.jpg'
    PlotSingleField(
        Dict['E_tensors'][EventNum],
        Dict['voltages'][EventNum],
        Dict['part_Zs'][EventNum],
        output_filename         = filename,
        step_index              = ii,
        EorB                    = 0,
        coil_center             = Dict['coil_centers'][EventNum],
        coil_direction          = Dict['coil_directions'][EventNum],
        part_speed              = Dict['part_speeds'][EventNum],
        comp_mins               = [Ex_min, Ey_min, Ez_min],
        comp_maxs               = [Ex_max, Ey_max, Ez_max],
        config                  = Dict['config'],
    )
    E_filelist.append(filename)
    # plot B field
    filename            = OutputDir+'/B_field_step'+str(ii)+'.jpg'
    PlotSingleField(
        Dict['B_tensors'][EventNum],
        Dict['voltages'][EventNum],
        Dict['part_Zs'][EventNum],
        output_filename         =filename,
        step_index              =ii,
        EorB                    =1,
        coil_center             =Dict['coil_centers'][EventNum],
        coil_direction          =Dict['coil_directions'][EventNum],
        part_speed              =Dict['part_speeds'][EventNum],
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
    filelist                = E_filelist,
    output_filename         = OutputDir+'/E_field.gif',
    duration                = 0.1
)
MakeGif(
    filelist                = B_filelist,
    output_filename         = OutputDir+'/B_field.gif',
    duration                = 0.1
)



