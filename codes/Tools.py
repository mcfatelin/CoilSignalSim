##############################################
## Tools for PlotField.py
## by Qing Lin
## @ 2021-11-23
###############################################
import imageio
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import gridspec
from matplotlib.colors import LogNorm
import pylab
import matplotlib as m
from matplotlib.patches import Ellipse

GlobalCMap              = plt.cm.viridis
GlobalSpeedOfLight      = 2.998e8 # m/s


def PlotSingleBField(FieldTensor, **kwargs):
    # load info
    output_filename     = kwargs.get('output_filename', 'test.jpg')
    step_index          = kwargs.get('step_index', 0)
    part_speed          = kwargs.get('part_speed', 0.01)
    comp_mins           = kwargs.get('comp_mins', [0, 0, 0])
    comp_maxs           = kwargs.get('comp_maxs', [1, 1, 1])
    config              = kwargs.get('config', {})
    ###########################
    ## construct canvas
    ############################
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
    fig = plt.figure(figsize=(20.0, 20.0))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    gs = gridspec.GridSpec(200, 150)
    gs.update(
        left=0.12,
        top=0.98,
        right=0.90,
        bottom=0.12,
        hspace=0.01,
        wspace=0.01,
    )
    axes = []
    for axis_id in range(3):
        xindex_lower = axis_id * 50 + 6
        xindex_upper = axis_id * 50 + 50 - 6
        ax_array = []
        for jj, _ in enumerate([-1, 0, 1, 2]):
            yindex_lower = jj * 50  + 6
            yindex_upper = jj * 50 + 50 - 6
            ax_array.append(plt.subplot(gs[yindex_lower:yindex_upper, xindex_lower:xindex_upper]))
        axes.append(ax_array)
    ############################
    ## Plot the tensor projections
    ############################
    world_box_size          = config.get('world_box_size', 500)
    cell_step               = config.get('cell_step', 5)
    num_cells               = int(np.floor(world_box_size / cell_step))
    bins                    = np.linspace(-0.5 * world_box_size, 0.5 * world_box_size, num_cells + 1)
    X, Y                    = np.meshgrid(bins, bins, indexing='ij')
    caxs                    = []
    coloraxes               = []
    for ii, ax_id in enumerate(range(3)):
        for jj, comp_id in enumerate([-1, 0, 1, 2]):
            axes[ii][jj], c = PlotField(
                X,
                Y,
                FieldTensor[step_index],
                axes[ii][jj],
                ax_id=ax_id,
                comp_id=comp_id,
                min=comp_mins[ii],
                max=comp_maxs[ii],
                box_size=world_box_size,
                cell_step=cell_step,
                coil_center=[0,0,0],
            )
            if ii == 0:
                Label = 'B[nT]'
                if comp_id == 0:
                    Label = 'Bx[nT]'
                elif comp_id == 1:
                    Label = 'By[nT]'
                elif comp_id == 2:
                    Label = 'Bz[nT]'
                cax = fig.add_axes([0.902, 0.78 - float(jj) * 0.213, 0.03, 0.17], label=Label)
                coloraxis = fig.colorbar(c, cax=cax)
                coloraxis.set_label(Label, fontsize=20)
                caxs.append(cax)
                coloraxes.append(coloraxis)
    # save
    plt.savefig(output_filename)
    plt.close()
    return


def PlotSingleField(FieldTensor, Voltages, Zs, **kwargs):
    # load info
    output_filename             = kwargs.get('output_filename', 'test.jpg')
    step_index                  = kwargs.get('step_index', 0)
    EorB                        = kwargs.get('EorB', 0)
    coil_center                 = kwargs.get('coil_center', [0,0,0])
    coil_direction              = kwargs.get('coil_direction', [0,0,1])
    part_speed                  = kwargs.get('part_speed', 0.01)
    comp_mins                   = kwargs.get('comp_mins', [0,0,0])
    comp_maxs                   = kwargs.get('comp_maxs', [1,1,1])
    config                      = kwargs.get('config', {})
    ###########################
    ## construct canvas
    ############################
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
    fig = plt.figure(figsize=(20.0, 25.0))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    gs = gridspec.GridSpec(250, 150)
    gs.update(
        left=0.12,
        top=0.98,
        right=0.90,
        bottom=0.12,
        hspace=0.01,
        wspace=0.01,
    )
    ax_voltage                  = plt.subplot(gs[:45,10:])
    axes                        = []
    for axis_id in range(3):
        xindex_lower            = axis_id*50 + 6
        xindex_upper            = axis_id*50 + 50 - 6
        ax_array                = []
        for jj, _ in enumerate([-1, 0, 1, 2]):
            yindex_lower        = jj*50 + 50 + 6
            yindex_upper        = jj*50 + 100 - 6
            ax_array.append(plt.subplot(gs[yindex_lower:yindex_upper, xindex_lower:xindex_upper]))
        axes.append(ax_array)
    ############################
    ## Plot the voltage vs. time
    ############################
    ax_voltage                  = PlotVoltageVsTime(
        Zs,
        Voltages,
        ax_voltage,
        part_speed              = part_speed,
        step_index              = step_index,
    )
    ############################
    ## Plot the tensor projections
    ############################
    world_box_size              = config.get('world_box_size', 500)
    cell_step                   = config.get('cell_step', 5)
    num_cells                   = int(np.floor(world_box_size / cell_step))
    bins                        = np.linspace(-0.5*world_box_size, 0.5*world_box_size, num_cells+1)
    X, Y                        = np.meshgrid(bins, bins, indexing='ij')
    caxs                        = []
    coloraxes                   = []
    for ii, ax_id in enumerate(range(3)):
        for jj, comp_id in enumerate([-1, 0, 1, 2]):
            axes[ii][jj], c     = PlotField(
                X,
                Y,
                FieldTensor[step_index],
                axes[ii][jj],
                ax_id           = ax_id,
                comp_id         = comp_id,
                min             = comp_mins[ii],
                max             = comp_maxs[ii],
                box_size        = world_box_size,
                cell_step       = cell_step,
                coil_center     = coil_center,
            )
            axes[ii][jj]        = PlotCoilCircle(
                coil_center,
                coil_direction,
                axes[ii][jj],
                ax_id           = ax_id,
                radius          = config.get('coil_diameter', 100)*0.5
            )
            if ii==0:
                # make color bar
                if EorB==0:
                    Label           = 'E[V/m]'
                    if comp_id==0:
                        Label       = 'Ex[V/m]'
                    elif comp_id==1:
                        Label       = 'Ey[V/m]'
                    elif comp_id==2:
                        Label       = 'Ez[V/m]'
                else:
                    Label = 'B[nT]'
                    if comp_id == 0:
                        Label = 'Bx[nT]'
                    elif comp_id == 1:
                        Label = 'By[nT]'
                    elif comp_id == 2:
                        Label = 'Bz[nT]'
                cax             = fig.add_axes([0.902, 0.653-float(jj)*0.171, 0.03, 0.13], label=Label)
                coloraxis       = fig.colorbar(c, cax=cax)
                coloraxis.set_label(Label, fontsize=20)
                caxs.append(cax)
                coloraxes.append(coloraxis)
    # save
    plt.savefig(output_filename)
    plt.close()
    return

def PlotVoltageVsTime(
        Zs, # in mm
        Voltages, # in V
        ax,
        **kwargs
):
    # load info
    part_speed                  = kwargs.get('part_speed', 0.01)
    step_index                  = kwargs.get('step_index', 0)
    # Get the times
    Ts                          = np.asarray(Zs) / (part_speed*GlobalSpeedOfLight)*1e-3*1e9 # in ns
    V_min                       = np.min(Voltages)
    V_max                       = np.max(Voltages)
    if V_max<=V_min:
        V_max                   = V_min + 1
    # Plot
    ax.plot(
        Ts,
        Voltages,
        ls          = 'solid',
        lw          = 3,
        color       = 'k',
    )
    # Draw line
    ax.plot(
        [Ts[step_index], Ts[step_index]],
        [V_min, V_max],
        ls          = 'solid',
        lw          = 3,
        color       = 'r'
    )
    # setting
    ax.set_xlim([Ts[0], Ts[-1]])
    ax.set_ylim([V_min, V_max])
    # ax.tick_params(axis='x', labelsize=20, width=2, length=5)
    # ax.tick_params(axis='y', labelsize=20, width=2, length=5)
    ax.set_xlabel('Time [ns]', fontsize=20)
    ax.set_ylabel('Voltage [V]', fontsize=20)
    return ax


def PlotField(
        X,
        Y,
        FieldTensor,
        ax,
        **kwargs
):
    # load info
    ax_id                   = kwargs.get('ax_id', 0)
    comp_id                 = kwargs.get('comp_id', -1)
    F_min                   = kwargs.get('min', 0)
    F_max                   = kwargs.get('max', F_min+1)
    # # debug
    # print("F_min = "+str(F_min))
    # print("F_max = "+str(F_max))
    box_size                = kwargs.get('box_size', 500)
    cell_step               = kwargs.get('cell_step', 5)
    coil_center             = kwargs.get('coil_center', [0,0,0])
    # Find the F at the requested plane
    target_axis_index       = int(np.floor( (coil_center[ax_id] + 0.5*box_size) / cell_step))
    if ax_id==0 and comp_id==-1:
        F                   = np.sqrt(np.sum(
            FieldTensor[:,target_axis_index,:,:]**2,
            axis            = 0
        ))
    elif ax_id==0:
        F                   = FieldTensor[comp_id, target_axis_index, :, :]
    elif ax_id==1 and comp_id==-1:
        F = np.sqrt(np.sum(
            FieldTensor[:, :, target_axis_index, :] ** 2,
            axis            =0
        ))
    elif ax_id == 1:
        F                   = FieldTensor[comp_id, :, target_axis_index, :]
    elif ax_id==2 and comp_id==-1:
        F                   = np.sqrt(np.sum(
            FieldTensor[:, :, :, target_axis_index] ** 2,
            axis            =0
        ))
    elif ax_id == 2:
        F                   = FieldTensor[comp_id, :, :, target_axis_index]
    # plot
    c = ax.pcolor(
        X,
        Y,
        F,
        vmin        = F_min,
        vmax        = F_max,
    )
    # setting
    ax.set_xlim([-0.5*box_size, 0.5*box_size])
    ax.set_ylim([-0.5*box_size, 0.5*box_size])
    # ax.tick_params(axis='x', labelsize=20, width=2, length=5)
    # ax.tick_params(axis='y', labelsize=20, width=2, length=5)
    if ax_id==0:
        ax.set_xlabel('Y [mm]', fontsize=20)
        ax.set_ylabel('Z [mm]', fontsize=20)
    elif ax_id==1:
        ax.set_xlabel('X [mm]', fontsize=20)
        ax.set_ylabel('Z [mm]', fontsize=20)
    else:
        ax.set_xlabel('X [mm]', fontsize=20)
        ax.set_ylabel('Y [mm]', fontsize=20)
    return ax, c

def PlotCoilCircle(
    coil_center,
    coil_direction,
    ax,
    **kwargs
):
    # load info
    ax_id               = kwargs.get('ax_id', 0)
    radius              = kwargs.get('radius', 50)
    # Calculate angle
    if ax_id==0:
        angle           = np.arctan2(coil_direction[2], coil_direction[1]) - 0.5*np.pi
    elif ax_id==1:
        angle           = np.arctan2(coil_direction[2], coil_direction[0]) - 0.5*np.pi
    else:
        angle           = np.arctan2(coil_direction[1], coil_direction[0]) - 0.5*np.pi
    angle               = angle/np.pi*180 # convert to degree
    # Calculate short-axis
    if ax_id==0:
        short_axis          = 2.*radius*np.sin(
            np.arctan2(coil_direction[0], np.sqrt(coil_direction[1]**2+coil_direction[2]**2))
        )
    elif ax_id==1:
        short_axis          = 2. * radius * np.sin(
            np.arctan2(coil_direction[1], np.sqrt(coil_direction[0] ** 2 + coil_direction[2] ** 2))
        )
    else:
        short_axis          = 2. * radius * np.sin(
            np.arctan2(coil_direction[2], np.sqrt(coil_direction[0] ** 2 + coil_direction[1] ** 2))
        )
    # Long axis
    long_axis               = 2.*radius
    # center
    if ax_id==0:
        center              = (coil_center[1], coil_center[2])
    elif ax_id==1:
        center              = (coil_center[0], coil_center[2])
    else:
        center              = (coil_center[0], coil_center[1])
    # plot
    ellipse                 = Ellipse(xy=center, width=long_axis, height=short_axis, angle=angle, lw=3, ls='solid', fill=False)
    ax.add_artist(ellipse)
    ellipse.set_facecolor("white")
    return ax



def MakeGif(**kwargs):
    # load info
    filelist                = kwargs.get('filelist', [])
    duration                = kwargs.get('duration', 0.2)
    output_filename         = kwargs.get('output_filename', 'test.gif')
    # make frames
    frames                  = []
    for image_name in filelist:
        frames.append(imageio.imread(image_name))
    imageio.mimsave(output_filename, frames, 'GIF', duration=duration)
    return