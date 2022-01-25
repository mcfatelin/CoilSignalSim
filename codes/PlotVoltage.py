#######################################################
## code for plotting the voltage waveforms
## by Qing Lin @ 2021-12-17
#######################################################
import numpy as np
import pickle as pkl
import sys
from copy import deepcopy





############################
## inputs
############################
if len(sys.argv)<3:
    Message             = 'python3 PlotVoltage.py <input pickle file> <output png file> <event ID> <(opt.) W or WO noise> <(opt.) number of coils>'
    print(Message)
    exit()


InputFilename           = sys.argv[1]
OutputFilename          = sys.argv[2]
EventID                 = int(sys.argv[3])
WithNoise               = False
if len(sys.argv)>4 and int(sys.argv[4])>0:
    WithNoise           = True
NumCoils                = 6
if len(sys.argv)>5:
    NumCoils            = int(sys.argv[5])


#########################
## Load file
#########################
Dict                    = pkl.load(open(InputFilename, 'rb'))

# load info
MaxNumEvents            = len(Dict['coil_ids'])
NumSamples              = None
for ii in range(MaxNumEvents):
    try:
        NumSamples      = len(Dict['voltages'][ii][0])
        break
    except:
        continue


if EventID>=MaxNumEvents:
    raise ValueError("Requested event out of range!")
if len(Dict['coil_ids'][EventID])==0:
    raise ValueError("No hit is found for this event!")

PartSpeed               = Dict['part_speeds'][EventID]

print("MaxNumEvents = "+str(MaxNumEvents))
print("NumSamples = "+str(NumSamples))
print("Particle speed = "+str(PartSpeed)+' c')
print("Particle center = "+str(Dict['part_centers'][EventID]))
print("Particle direction = "+str(Dict['part_directions'][EventID]))

##########################
## Construct waveform
##########################
sample_size             = Dict['sample_sizes'][EventID]
min_hit_time            = np.min(Dict['hit_times'][EventID])
max_hit_time            = np.max(Dict['hit_times'][EventID])
time_range              = (
    min_hit_time - NumSamples*sample_size/2,
    max_hit_time + NumSamples*sample_size/2.
)

voltages                = np.asarray(Dict['voltages_wo_noise'][EventID])
if WithNoise:
    print("With noise!")
    voltages            = np.asarray(Dict['voltages'][EventID])
# voltages[:,:3]          = 0
max_voltage             = np.max(voltages)
wfs                     = []
for jj in range(len(Dict['hit_times'][EventID])):
    SingleWF            = {}
    coil_id             = int(Dict['coil_ids'][EventID][jj])
    mean_hit_time       = Dict['hit_times'][EventID][jj]
    bins                = np.linspace(
        mean_hit_time - float(NumSamples)*sample_size/2.,
        mean_hit_time + float(NumSamples)*sample_size/2.,
        NumSamples + 1
    )
    centers             = 0.5*(bins[1:] + bins[:-1])
    SingleWF['coil_id'] = coil_id
    SingleWF['t']       = centers
    SingleWF['v']       = voltages[jj] + float(jj)*max_voltage # for debug
    wfs.append(deepcopy(SingleWF))



#########################
## Plot
#########################
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import gridspec
from matplotlib.colors import LogNorm
import pylab
import matplotlib as m
from matplotlib.patches import Ellipse

# construct figure
params = {
    'backend': 'Agg',
    # colormap
    'image.cmap': 'viridis',
    # figure
    'figure.figsize': (9, 6),
    'font.size': 32,
    'font.family': 'serif',
    'font.serif': ['Times'],
    # axes
    'axes.titlesize': 42,
    'axes.labelsize': 32,
    'axes.linewidth': 2,
    # ticks
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'xtick.major.size': 16,
    'xtick.minor.size': 8,
    'ytick.major.size': 16,
    'ytick.minor.size': 8,
    'xtick.major.width': 2,
    'xtick.minor.width': 2,
    'ytick.major.width': 2,
    'ytick.minor.width': 2,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    # markers
    'lines.markersize': 2,
    'lines.markeredgewidth': 1,
    'errorbar.capsize': 8,
    'lines.linewidth': 1,
    'lines.linestyle': 'solid',
    #    'lines.marker' : None,
    'savefig.bbox': 'tight',
    'legend.fontsize': 24,
    # 'legend.fontsize': 18,
    # 'figure.figsize': (15, 5),
    # 'axes.labelsize': 18,
    # 'axes.titlesize':18,
    # 'xtick.labelsize':14,
    # 'ytick.labelsize':14
    'axes.labelsize': 24,
    'axes.titlesize': 24,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    #     'mathtext.fontset': 'dejavuserif'
    'mathtext.fontset': 'cm'
}
plt.rcParams.update(params)
fig = plt.figure(figsize=(8.0, 10.0))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
gs = gridspec.GridSpec(100, 100)
gs.update(
    left=0.12,
    top=0.92,
    right=0.90,
    bottom=0.12,
    hspace=0.01,
    wspace=0.01,
)


ax          = plt.subplot(gs[:,:])

GlobalCMap          = plt.cm.viridis
# plot and setting
for ii in range(len(wfs)):
    ax.plot(
        wfs[ii]['t'],
        wfs[ii]['v'],
        color           = GlobalCMap(float(ii)/float(NumCoils)),
        ls              = 'solid',
        lw              = 3.,
        label           = "Coil ID = "+str(wfs[ii]['coil_id'])
    )


# setting
ax.set_xlim([time_range[0], time_range[1]])
# ax.set_ylim([V_min, V_max])
ax.tick_params(axis='x', labelsize=20, width=2, length=5)
ax.tick_params(axis='y', labelsize=20, width=2, length=5)
ax.set_xlabel('Time [ns]', fontsize=20)
ax.set_ylabel('Voltage [V]', fontsize=20)
ax.legend(
    loc                 ='best',
    fontsize            = 10,
    ncol                = 2
)
plt.show()


# temporary output
OutputDict              = {}
print("output coil_id="+str(wfs[4]['coil_id']))
OutputDict['t']         = wfs[4]['t']
OutputDict['v']         = wfs[4]['v']
pkl.dump(
    OutputDict,
    open("ToYe.pkl",'wb')
)




