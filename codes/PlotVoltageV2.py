#############################################
## Plot the voltage with samples times now recorded
## by Qing Lin @ 2022-03-28
#############################################
import numpy as np
import pickle as pkl
import sys


##############################
## Input
##############################
if len(sys.argv)<2:
    print("python3 PlotVoltageV2.py <pkl file> <Event Number> < (opt.) Output filename> <(opt.) Whether to plot RLC waveforms>")
    exit()


InputFilename                   = sys.argv[1]
EventNum                        = int(sys.argv[2])
OutputFilename                  = None
if len(sys.argv)>3:
    OutputFilename              = sys.argv[3]
RLCPlotting                     = False
if len(sys.argv)>4 and int(sys.argv[4])>0:
    RLCPlotting                 = True


#############################
## Open file
#############################

Dict                            = pkl.load(open(InputFilename, 'rb'))

#############################
## Print out info
#############################
print("===>>> Part. start point: "+str(Dict['part_centers'][EventNum]))
print("===>>> Part. direction: "+str(Dict['part_directions'][EventNum]))
print("===>>> Part. speed: "+str(Dict['part_speeds'][EventNum]))

############################
## Dump variables
############################


coil_ids                        = Dict['coil_ids'][EventNum]
NumCoils                        = len(coil_ids)
wfs                             = []
Ts                              = []


for hit_time, voltages, sample_times in zip(
        Dict['hit_times'][EventNum],
        Dict['voltages'][EventNum],
        Dict['sample_times'][EventNum]
):
    wfs.append(voltages)
    sample_times                = np.asarray(sample_times)
    Ts.append(sample_times+hit_time)

wfs                              = np.asarray(wfs)
Ts                               = np.asarray(Ts)

if RLCPlotting:
    wfs_RLC_sig                     = []
    wfs_RLC_noise                   = []
    for voltages_sig, voltages_noise in zip(
        Dict['voltages_RLC'][EventNum],
        Dict['noises_RLC'][EventNum],
    ):
        # debug
        # print("voltages_sig = "+str(voltages_sig.real))
        # print("voltages_noise = "+str(voltages_noise.real))
        wfs_RLC_sig.append(voltages_sig.real)
        wfs_RLC_noise.append(voltages_noise.real)
    wfs_RLC_sig                     = np.asarray(wfs_RLC_sig)
    wfs_RLC_noise                   = np.asarray(wfs_RLC_noise)

#############################
## calculate amp and T range
#############################
amp_max                         = np.max(wfs)
amp_min                         = np.min(wfs)
T_max                           = np.max(Ts)
T_min                           = np.min(Ts)

if RLCPlotting:
    amp_max                         = np.max(wfs_RLC_sig+wfs_RLC_noise)
    amp_min                         = np.min(wfs_RLC_sig+wfs_RLC_noise)



#############################
## Plot
#############################
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
if not RLCPlotting:
    for ii in range(len(wfs)):
        coil_id             = coil_ids[ii]
        ax.plot(
            Ts[ii],
            wfs[ii]+1.1*float(coil_id)*(amp_max-amp_min),
            color           = GlobalCMap(float(coil_id)/float(NumCoils)),
            ls              = 'solid',
            lw              = 3.,
            label           = "Coil ID = "+str(coil_id)
        )

if RLCPlotting:
    for ii in range(len(wfs)):
        coil_id = coil_ids[ii]
        ax.plot(
            Ts[ii],
            # wfs_RLC_sig[ii]+wfs_RLC_noise[ii] + 1.1 * float(coil_id) * (amp_max - amp_min),
            wfs_RLC_sig[ii] + 1.1 * float(coil_id) * (amp_max - amp_min),
            color=GlobalCMap(float(coil_id) / float(NumCoils)),
            ls='dashed',
            lw=3.,
            label="RLC sig Coil ID = " + str(coil_id)
        )
        ax.plot(
            Ts[ii],
            # wfs_RLC_sig[ii]+wfs_RLC_noise[ii] + 1.1 * float(coil_id) * (amp_max - amp_min),
            wfs_RLC_noise[ii] + 1.1 * float(coil_id) * (amp_max - amp_min),
            color=GlobalCMap(float(coil_id) / float(NumCoils)),
            ls='dotted',
            lw=3.,
            label="RLC noise Coil ID = " + str(coil_id)
        )
        ax.plot(
            Ts[ii],
            wfs_RLC_sig[ii]+wfs_RLC_noise[ii] + 1.1 * float(coil_id) * (amp_max - amp_min),
            # wfs_RLC_sig[ii] + 1.1 * float(coil_id) * (amp_max - amp_min),
            color=GlobalCMap(float(coil_id) / float(NumCoils)),
            ls='solid',
            lw=3.,
            label="RLC total Coil ID = " + str(coil_id)
        )


# setting
ax.set_xlim([T_min, T_max])
ax.set_ylim([-np.abs(amp_min), 1.2*float(NumCoils)*(amp_max-amp_min)+amp_max])
ax.tick_params(axis='x', labelsize=20, width=2, length=5)
ax.tick_params(axis='y', labelsize=20, width=2, length=5)
ax.set_xlabel('Time [ns]', fontsize=20)
if not RLCPlotting:
    ax.set_ylabel('Voltage [V]', fontsize=20)
else:
    ax.set_ylabel('Current [A]', fontsize=20)
ax.legend(
    loc                 ='best',
    fontsize            = 10,
    ncol                = 3
)

if OutputFilename is not None:
    plt.savefig(OutputFilename)
plt.show()


# temporary output
# OutputDict              = {}
# print("output coil_id="+str(wfs[4]['coil_id']))
# OutputDict['t']         = wfs[4]['t']
# OutputDict['v']         = wfs[4]['v']
# pkl.dump(
#     OutputDict,
#     open("ToYe.pkl",'wb')
# )

