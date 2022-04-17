################################################
## Plot the response function
################################################
import pickle as pkl
import numpy as np
import sys

########################
## hardcoded input
########################
Filename                    = sys.argv[1]
OutputFilename              = sys.argv[2]
Mode                        = int(sys.argv[3])




Dict                        = pkl.load(open(Filename, 'rb'))


if Mode==0:
    freq                        = np.asarray(Dict['signal']['f'])
    amp                         = np.asarray(Dict['signal']['amp'])
    phase_x                     = np.asarray(Dict['signal']['phase_x'])
    phase_y                     = np.asarray(Dict['signal']['phase_y'])
    theta                       = np.arctan2(phase_y, phase_x)
else:
    freq = np.asarray(Dict['noises'][-1]['f'])
    amp = np.asarray(Dict['noises'][-1]['amp'])
    phase_x = np.asarray(Dict['noises'][-1]['phase_x'])
    phase_y = np.asarray(Dict['noises'][-1]['phase_y'])
    theta = np.arctan2(phase_y, phase_x)






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
fig = plt.figure(figsize=(16.0, 12.0))
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


ax1         = plt.subplot(gs[:45, :45])
ax3         = plt.subplot(gs[55:, :45])
ax2         = plt.subplot(gs[:45, 55:])
ax4         = plt.subplot(gs[55:, 55:])



ax1.plot(
    freq,
    amp,
    c               = 'k',
    ls              = 'solid',
    lw              = 3.,
)
ax2.plot(
    freq,
    theta,
    c               = 'k',
    ls              = 'solid',
    lw              = 3.,
)
ax3.plot(
    freq,
    phase_x,
    c               = 'k',
    ls              = 'solid',
    lw              = 3.,
)
ax4.plot(
    freq,
    phase_y,
    c               = 'k',
    ls              = 'solid',
    lw              = 3.,
)


# f_min           = freq[0]
# f_max           = freq[-1]

f_min           = 1e-6
f_max           = 1.

# f_min           = 1e-4
# f_max           = 1e-2

# setting
ax1.set_xlim([f_min, f_max])
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.tick_params(axis='x', labelsize=20, width=2, length=5)
ax1.tick_params(axis='y', labelsize=20, width=2, length=5)
ax1.set_xlabel('Freq [MHz]', fontsize=20)
ax1.set_ylabel('Amp', fontsize=20)

ax2.set_xlim([f_min, f_max])
ax2.set_xscale('log')
ax2.tick_params(axis='x', labelsize=20, width=2, length=5)
ax2.tick_params(axis='y', labelsize=20, width=2, length=5)
ax2.set_xlabel('Freq [MHz]', fontsize=20)
ax2.set_ylabel('Theta [rad]', fontsize=20)


ax3.set_xlim([f_min, f_max])
ax3.set_xscale('log')
ax3.tick_params(axis='x', labelsize=20, width=2, length=5)
ax3.tick_params(axis='y', labelsize=20, width=2, length=5)
ax3.set_xlabel('Freq [MHz]', fontsize=20)
ax3.set_ylabel('cos(theta)', fontsize=20)

ax4.set_xlim([f_min, f_max])
ax4.set_xscale('log')
ax4.tick_params(axis='x', labelsize=20, width=2, length=5)
ax4.tick_params(axis='y', labelsize=20, width=2, length=5)
ax4.set_xlabel('Freq [MHz]', fontsize=20)
ax4.set_ylabel('sin(theta)', fontsize=20)

plt.savefig(OutputFilename)
# plt.savefig("Response_zoom.png")
plt.show()



