#############################################################
## Quickly plot voltage vs time
#############################################################

import pickle as pkl
import numpy as np
import sys

SpeedOfLight                = 2.998e8 # m/s

#################################
## Inputs
#################################
if len(sys.argv)<2:
    print("python3 PlotField.py <input File> <Event Number>")
    exit()

InputFilename               = sys.argv[1]
EventNum                    = int(sys.argv[2])

################################
## Load
################################
Dict                        = pkl.load(open(InputFilename, 'rb'))


if EventNum>=len(Dict['voltages']):
    raise ValueError("Requested event number is not available!")

###############################
## Plot
##############################
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
fig = plt.figure(figsize=(10.0, 6.0))
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

ax                  = plt.subplot(gs[:,:])

part_speed          = Dict['part_speeds'][EventNum]*SpeedOfLight

ax.plot(
    np.asarray(Dict['part_Zs'][EventNum])/part_speed*1e6,
    np.asarray(Dict['voltages'][EventNum]),
    ls              = 'solid',
    lw              = 3,
    color           ='k',
)


# setting
# ax.set_xlim([Ts[0], Ts[-1]])
# ax.set_ylim([V_min, V_max])
# ax.tick_params(axis='x', labelsize=20, width=2, length=5)
# ax.tick_params(axis='y', labelsize=20, width=2, length=5)
ax.set_xlabel('Time [ns]', fontsize=20)
ax.set_ylabel('Voltage [V]', fontsize=20)
plt.show()
