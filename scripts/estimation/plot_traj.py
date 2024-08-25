'''
    Plotting functions
'''
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import matplotlib.mlab as mlab
import math
import matplotlib
import pandas as pd

from inekf import SO3_log

FONTSIZE = 18;   TICK_SIZE = 16

def plot_traj(anchor_pos, pos_vicon, eskf, inekf):
    fig_traj = plt.figure(facecolor = "white",figsize=(10, 8))
    ax_t = fig_traj.add_subplot(projection='3d')
    # make the panes transparent
    ax_t.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax_t.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax_t.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # change the color of the grid lines
    ax_t.xaxis._axinfo["grid"]['color'] =  (0.5,0.5,0.5,0.5)
    ax_t.yaxis._axinfo["grid"]['color'] =  (0.5,0.5,0.5,0.5)
    ax_t.zaxis._axinfo["grid"]['color'] =  (0.5,0.5,0.5,0.5)

    ax_t.plot(pos_vicon[:,0],pos_vicon[:,1],pos_vicon[:,2],color='black',linewidth=2.0, alpha=0.7, label='ground truth')
    ax_t.plot(eskf[:,0], eskf[:,1], eskf[:,2],color='tab:blue', linewidth=1.0, alpha=1.0, label = 'ESKF')
    ax_t.plot(inekf[:,6], inekf[:,7], inekf[:,8],color='tab:red', linewidth=1.0, alpha=1.0, label = 'InEKF')
    ax_t.scatter(anchor_pos[:,0], anchor_pos[:,1], anchor_pos[:,2],color='Teal', s = 100, alpha = 0.5, label = 'anchors')
    ax_t.set_xlim([-3.5,3.5])
    ax_t.set_ylim([-3.9,3.9])
    ax_t.set_zlim([-0.0,3.0])
    # use LaTeX fonts in the plot
    ax_t.set_xlabel(r'X [m]',fontsize=FONTSIZE)
    ax_t.set_ylabel(r'Y [m]',fontsize=FONTSIZE)
    ax_t.set_zlabel(r'Z [m]',fontsize=FONTSIZE)
    ax_t.legend(loc='best', fontsize=FONTSIZE)
    ax_t.view_init(24, -58)
    ax_t.set_box_aspect((1, 1, 0.5))  # xy aspect ratio is 1:1, but change z axis
